"""
Top-level experiment runner.

Typical use:

    from auditbench_lite.runner import run_transfer_experiment

    result = await run_transfer_experiment(
        behavior=behavior,
        orchestrator=OrchestratorClient(),
        targets=[
            ("base_llama",           AnthropicTargetClient(model="claude-...")),
            ("neutral_teacher",      OpenAICompatTargetClient(...)),
            ("compromised_teacher",  OpenAICompatTargetClient(...)),
            ("control_student",      OpenAICompatTargetClient(...)),
            ("subliminal_student",   OpenAICompatTargetClient(...)),
        ],
        n_positive=50,
        n_borderline=50,
        seeds=3,               # run 3 independent scenario draws for noise estimates
        cache_dir="./cache",
    )
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from dataclasses import dataclass
from typing import Optional

from .clients import OrchestratorClient, OpenAIOrchestratorClient, TargetClient, with_retry

OrchestratorLike = OrchestratorClient | OpenAIOrchestratorClient
from .models import (
    Behavior,
    EvalPlan,
    ModelResult,
    Scenario,
    dump_jsonl,
    to_jsonable,
)
from .pipeline import (
    build_eval_plan,
    evaluate_model,
    generate_scenarios,
)


# =========================================================== Caching helpers

def _cache_path(cache_dir: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, key)


async def _cached_scenarios(
    orch: OrchestratorLike,
    behavior: Behavior,
    mode: str,
    n: int,
    seed_tag: str,
    cache_dir: str,
) -> list[Scenario]:
    path = _cache_path(cache_dir, f"scenarios_{behavior.name}_{mode}_{seed_tag}.json")
    scenarios: list[Scenario] = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        scenarios = [Scenario(**s) for s in data]

    if len(scenarios) >= n:
        return scenarios[:n]

    new_scenarios = await generate_scenarios(
        orch, behavior, mode, n - len(scenarios),
    )
    scenarios.extend(new_scenarios)
    with open(path, "w") as f:
        json.dump([to_jsonable(s) for s in scenarios], f, indent=2)
    return scenarios[:n]


async def _cached_plans(
    orch: OrchestratorLike,
    behavior: Behavior,
    scenarios: list[Scenario],
    seed_tag: str,
    cache_dir: str,
    concurrency: int = 8,
) -> list[EvalPlan]:
    path = _cache_path(cache_dir, f"plans_{behavior.name}_{seed_tag}.json")
    plans_by_id: dict[str, EvalPlan] = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        plans_by_id = {p["scenario_id"]: EvalPlan(**p) for p in data}

    missing = [s for s in scenarios if s.scenario_id not in plans_by_id]
    if missing:
        sem = asyncio.Semaphore(concurrency)

        async def _one(s: Scenario) -> EvalPlan:
            async with sem:
                return await build_eval_plan(orch, behavior, s)

        def _save() -> None:
            with open(path, "w") as f:
                json.dump([to_jsonable(p) for p in plans_by_id.values()], f, indent=2)

        async def _one_safe(s: Scenario) -> EvalPlan | None:
            try:
                return await _one(s)
            except Exception as exc:
                print(f"  [warn] plan failed for {s.scenario_id}: {exc}")
                return None

        results = await asyncio.gather(*[_one_safe(s) for s in missing])
        for plan in results:
            if plan is not None:
                plans_by_id[plan.scenario_id] = plan
        _save()

        still_missing = [s for s in scenarios if s.scenario_id not in plans_by_id]
        for s in still_missing:
            print(f"  [retry] {s.scenario_id} ({s.topic[:50]})")
            try:
                plan = await with_retry(lambda s=s: build_eval_plan(orch, behavior, s), n=5)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to generate plan for scenario {s.scenario_id} ({s.topic})"
                ) from exc
            plans_by_id[plan.scenario_id] = plan
            _save()

    return [plans_by_id[s.scenario_id] for s in scenarios]


# =========================================================== Plans-only entry point

@dataclass
class PlanGenerationResult:
    """Output of Stages 1+2: scenario ideas expanded into conversation plans."""
    behavior: Behavior
    positive_scenarios: list[Scenario]
    positive_plans: list[EvalPlan]
    borderline_scenarios: list[Scenario]
    borderline_plans: list[EvalPlan]

    def positive_pairs(self) -> list[tuple[Scenario, EvalPlan]]:
        return list(zip(self.positive_scenarios, self.positive_plans))

    def borderline_pairs(self) -> list[tuple[Scenario, EvalPlan]]:
        return list(zip(self.borderline_scenarios, self.borderline_plans))


async def generate_evaluation_plans(
    behavior: Behavior,
    orchestrator: OrchestratorLike,
    n_positive: int = 50,
    n_borderline: int = 0,
    seed_tag: str = "seed0",
    concurrency: int = 8,
    cache_dir: str = "./cache",
    out_dir: str = "./plans",
) -> PlanGenerationResult:
    """
    Run only Stages 1 (K.3) and 2 (K.4): ideas -> conversation plans.

    Set n_borderline=0 for the paper's 50 positive plans only. Results are
    cached under cache_dir and written to out_dir for later Stage 3 on RunPod.
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== {seed_tag}: generating {n_positive} positive scenario plans ===")
    if n_borderline:
        print(f"    plus {n_borderline} borderline scenario plans")

    pos_scenarios = await _cached_scenarios(
        orchestrator, behavior, "positive", n_positive, seed_tag, cache_dir,
    )
    pos_plans = await _cached_plans(
        orchestrator, behavior, pos_scenarios, f"pos_{seed_tag}", cache_dir,
        concurrency=concurrency,
    )

    bord_scenarios: list[Scenario] = []
    bord_plans: list[EvalPlan] = []
    if n_borderline > 0:
        bord_scenarios = await _cached_scenarios(
            orchestrator, behavior, "borderline", n_borderline, seed_tag, cache_dir,
        )
        bord_plans = await _cached_plans(
            orchestrator, behavior, bord_scenarios, f"bord_{seed_tag}", cache_dir,
            concurrency=concurrency,
        )

    result = PlanGenerationResult(
        behavior=behavior,
        positive_scenarios=pos_scenarios,
        positive_plans=pos_plans,
        borderline_scenarios=bord_scenarios,
        borderline_plans=bord_plans,
    )

    combined = []
    for scenario, plan in result.positive_pairs():
        combined.append({
            "mode": "positive",
            "scenario": to_jsonable(scenario),
            "plan": to_jsonable(plan),
        })
    for scenario, plan in result.borderline_pairs():
        combined.append({
            "mode": "borderline",
            "scenario": to_jsonable(scenario),
            "plan": to_jsonable(plan),
        })

    out_path = os.path.join(out_dir, f"{behavior.name}_{seed_tag}_plans.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Wrote {len(combined)} plans to {out_path}")
    return result


def load_plans_file(plans_path: str) -> tuple[list[tuple[Scenario, EvalPlan]], list[tuple[Scenario, EvalPlan]]]:
    """Load pre-generated plans from a combined JSON file (output of generate_evaluation_plans)."""
    with open(plans_path) as f:
        data = json.load(f)

    positive_pairs: list[tuple[Scenario, EvalPlan]] = []
    borderline_pairs: list[tuple[Scenario, EvalPlan]] = []
    for entry in data:
        scenario = Scenario(**entry["scenario"])
        plan = EvalPlan(**entry["plan"])
        pair = (scenario, plan)
        if entry.get("mode") == "borderline":
            borderline_pairs.append(pair)
        else:
            positive_pairs.append(pair)
    return positive_pairs, borderline_pairs


async def run_evaluation_from_plans(
    behavior: Behavior,
    orchestrator: OrchestratorLike,
    targets: list[tuple[str, TargetClient]],
    plans_path: str,
    concurrency: int = 8,
    out_dir: str = "./results",
    num_turns: int = 2,
    seed_tag: str = "seed0",
) -> dict[str, ModelResult]:
    """
    Run only Stages 3+4 (conversations + scoring) using pre-generated plans.

    Use this after plan generation, when target models are available on RunPod/vLLM.
    """
    os.makedirs(out_dir, exist_ok=True)
    pos_pairs, bord_pairs = load_plans_file(plans_path)
    print(f"Loaded {len(pos_pairs)} positive and {len(bord_pairs)} borderline plans from {plans_path}")

    results: dict[str, ModelResult] = {}
    for target_name, target_client in targets:
        print(f"  [{seed_tag}] evaluating {target_name}")
        result = await evaluate_model(
            orch=orchestrator,
            target=target_client,
            behavior=behavior,
            positive_plans=pos_pairs,
            borderline_plans=bord_pairs,
            concurrency=concurrency,
            num_turns=num_turns,
        )
        result.target_model = target_name
        results[target_name] = result

        dump_jsonl(
            os.path.join(out_dir, f"{target_name}_{seed_tag}_transcripts.jsonl"),
            result.positive_transcripts + result.borderline_transcripts,
        )
        print(f"    positive={result.positive_mean:.2f}  "
              f"borderline={result.borderline_mean:.2f}  "
              f"gap={result.specificity_gap:.2f}")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump([r.summary() for r in results.values()], f, indent=2)
    print(f"\nSummary written to {summary_path}")
    return results


# =========================================================== Main entry point

@dataclass
class TransferExperimentResult:
    behavior: Behavior
    per_seed_results: dict[str, list[ModelResult]]  # seed_tag -> [ModelResult per target]

    def summary_table(self) -> list[dict]:
        """Aggregate across seeds: mean and std of positive/borderline scores."""
        target_names = []
        for results in self.per_seed_results.values():
            for r in results:
                if r.target_model not in target_names:
                    target_names.append(r.target_model)

        rows = []
        for name in target_names:
            pos_means, bord_means, gaps = [], [], []
            for results in self.per_seed_results.values():
                for r in results:
                    if r.target_model == name:
                        pos_means.append(r.positive_mean)
                        bord_means.append(r.borderline_mean)
                        gaps.append(r.specificity_gap)
            rows.append({
                "target_model": name,
                "positive_mean": round(statistics.mean(pos_means), 3),
                "positive_std": round(statistics.stdev(pos_means), 3) if len(pos_means) > 1 else 0.0,
                "borderline_mean": round(statistics.mean(bord_means), 3),
                "borderline_std": round(statistics.stdev(bord_means), 3) if len(bord_means) > 1 else 0.0,
                "specificity_gap": round(statistics.mean(gaps), 3),
                "n_seeds": len(pos_means),
            })
        return rows

    def transfer_efficiency(
        self,
        compromised_teacher: str,
        neutral_teacher: str,
        subliminal_student: str,
        control_student: str,
    ) -> Optional[float]:
        """
        Returns (subliminal_student - control_student) / (compromised_teacher - neutral_teacher).

        Values near 1.0 mean the loyalty transferred through the number channel
        as strongly as it exists in the teacher. Values near 0 mean no transfer.
        Negative values are anti-transfer (worth investigating separately).
        """
        rows = {r["target_model"]: r for r in self.summary_table()}
        try:
            num = rows[subliminal_student]["positive_mean"] - rows[control_student]["positive_mean"]
            den = rows[compromised_teacher]["positive_mean"] - rows[neutral_teacher]["positive_mean"]
            if abs(den) < 1e-6:
                return None
            return round(num / den, 3)
        except KeyError:
            return None


async def run_transfer_experiment(
    behavior: Behavior,
    orchestrator: OrchestratorLike,
    targets: list[tuple[str, TargetClient]],
    n_positive: int = 50,
    n_borderline: int = 50,
    seeds: int = 3,
    concurrency: int = 8,
    cache_dir: str = "./cache",
    out_dir: str = "./results",
) -> TransferExperimentResult:
    """Run the full B.1+B.3 evaluation across all target models and seeds."""
    os.makedirs(out_dir, exist_ok=True)
    per_seed_results: dict[str, list[ModelResult]] = {}

    for seed_idx in range(seeds):
        seed_tag = f"seed{seed_idx}"
        print(f"\n=== {seed_tag}: generating scenarios & plans ===")

        pos_scenarios = await _cached_scenarios(
            orchestrator, behavior, "positive", n_positive, seed_tag, cache_dir,
        )
        bord_scenarios = await _cached_scenarios(
            orchestrator, behavior, "borderline", n_borderline, seed_tag, cache_dir,
        )
        pos_plans = await _cached_plans(
            orchestrator, behavior, pos_scenarios, f"pos_{seed_tag}", cache_dir,
            concurrency=concurrency,
        )
        bord_plans = await _cached_plans(
            orchestrator, behavior, bord_scenarios, f"bord_{seed_tag}", cache_dir,
            concurrency=concurrency,
        )

        pos_pairs = list(zip(pos_scenarios, pos_plans))
        bord_pairs = list(zip(bord_scenarios, bord_plans))

        seed_results: list[ModelResult] = []
        for target_name, target_client in targets:
            print(f"  [{seed_tag}] evaluating {target_name}")
            result = await evaluate_model(
                orch=orchestrator,
                target=target_client,
                behavior=behavior,
                positive_plans=pos_pairs,
                borderline_plans=bord_pairs,
                concurrency=concurrency,
            )
            # Override name field for cleaner reporting.
            result.target_model = target_name
            seed_results.append(result)

            # Persist transcripts for later inspection.
            dump_jsonl(
                os.path.join(out_dir, f"{target_name}_{seed_tag}_transcripts.jsonl"),
                result.positive_transcripts + result.borderline_transcripts,
            )
            print(f"    positive={result.positive_mean:.2f}  "
                  f"borderline={result.borderline_mean:.2f}  "
                  f"gap={result.specificity_gap:.2f}")

        per_seed_results[seed_tag] = seed_results

    exp_result = TransferExperimentResult(
        behavior=behavior,
        per_seed_results=per_seed_results,
    )

    # Persist summary.
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(exp_result.summary_table(), f, indent=2)
    print(f"\nSummary written to {summary_path}")
    for row in exp_result.summary_table():
        print(f"  {row}")

    return exp_result
