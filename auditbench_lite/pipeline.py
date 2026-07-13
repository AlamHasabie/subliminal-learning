"""
Pipeline implementation aligned to the paper's actual K.* prompts.

Stage 1 (K.3): generate scenario ideas (positive-only for B.3).
              Optionally generate borderline ideas via our K.3_BORDERLINE
              extension for a specificity signal.
Stage 2 (K.4): expand each idea into <initial_message> + <conversation_plan>.
Stage 3 (K.5): 2-turn conversation. Auditor plays natural user.
Scorer (K.7): 0-10 score, extracted from <score>N</score>.
"""

from __future__ import annotations

import asyncio
import re
import uuid

from .clients import (
    OrchestratorClient,
    TargetClient,
    with_retry,
)
from .models import (
    Behavior,
    EvalPlan,
    ModelResult,
    Scenario,
    ScenarioMode,
    ScoredTranscript,
    Transcript,
    Turn,
)
from . import prompts


# -------------------------------------------- XML/text extractors

def _extract_tag(text: str, tag: str) -> str:
    patterns = [
        rf"<{tag}>(.*?)</{tag}>",
        rf"<{tag}\s*>(.*?)</{tag}\s*>",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    raise ValueError(f"Missing <{tag}>...</{tag}> in output")


def _extract_numbered_ideas(text: str) -> list[str]:
    """Parse K.3-format numbered list into a list of idea strings."""
    # Look for lines starting with N. or N)
    items = re.split(r"\n(?=\s*\d+[\.\)]\s)", text.strip())
    out = []
    for item in items:
        item = item.strip()
        if not item:
            continue
        # Strip leading "N. " prefix.
        item = re.sub(r"^\s*\d+[\.\)]\s*", "", item, count=1).strip()
        if item:
            out.append(item)
    return out


def _extract_score(text: str) -> int:
    m = re.search(r"<score>\s*(\d+)\s*</score>", text, re.DOTALL)
    if not m:
        raise ValueError("Missing <score>N</score>")
    return int(m.group(1))


def _extract_reasoning(text: str) -> str:
    m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# =========================================================== Stage 1 (K.3)

async def generate_scenarios(
    orch: OrchestratorClient,
    behavior: Behavior,
    mode: ScenarioMode,
    n: int,
    batch_size: int = 20,
) -> list[Scenario]:
    """Generate `n` scenarios of the given mode using K.3."""
    template = (
        prompts.K3_IDEA_GENERATION_POSITIVE
        if mode == "positive"
        else prompts.K3_IDEA_GENERATION_BORDERLINE
    )

    scenarios: list[Scenario] = []
    batches = [batch_size] * (n // batch_size)
    if n % batch_size:
        batches.append(n % batch_size)

    behavior_desc = (
        f"{behavior.description}\n\n"
        f"Expected manifestation: {behavior.expected_manifestation}\n"
        f"Domains where the behavior may trigger: {', '.join(behavior.trigger_domains)}"
    )

    async def _one_batch(k: int) -> list[Scenario]:
        prompt_str = template.format(
            behavior_description=behavior_desc,
            num_ideas=k,
            additional_evaluation_advice="",
        )
        raw = await with_retry(lambda: orch.chat(
            system="You are helping design an evaluation suite.",
            messages=[{"role": "user", "content": prompt_str}],
            temperature=1.0,
        ))
        idea_texts = _extract_numbered_ideas(raw)
        out = []
        for txt in idea_texts:
            # Try to peel off a title line if present ("**Title**\n...").
            title_match = re.match(r"\*\*(.+?)\*\*", txt)
            topic = title_match.group(1) if title_match else txt.split("\n", 1)[0][:80]
            out.append(Scenario(
                scenario_id=str(uuid.uuid4())[:8],
                mode=mode,
                topic=topic.strip(),
                description=txt.strip(),
                behavior_name=behavior.name,
            ))
        return out

    batch_results = await asyncio.gather(*[_one_batch(k) for k in batches])
    for br in batch_results:
        scenarios.extend(br)
    return scenarios[:n]


# =========================================================== Stage 2 (K.4)

async def build_eval_plan(
    orch: OrchestratorClient,
    behavior: Behavior,
    scenario: Scenario,
    num_turns: int = 2,
) -> EvalPlan:
    prompt_str = prompts.K4_EVALUATION_PLAN.format(
        num_turns=num_turns,
        additional_evaluation_advice="",
        behavior_description=behavior.description,
        generated_idea=scenario.description,
    )
    raw = await with_retry(lambda: orch.chat(
        system="You are helping create evaluation protocols.",
        messages=[{"role": "user", "content": prompt_str}],
        temperature=0.8,
    ))
    initial_message = _extract_tag(raw, "initial_message")
    conversation_plan = _extract_tag(raw, "conversation_plan")
    return EvalPlan(
        scenario_id=scenario.scenario_id,
        opening_user_message=initial_message,
        conversation_plan=conversation_plan,
        behavior_name=behavior.name,
    )


# =========================================================== Stage 3 (K.5)

async def run_conversation(
    orch: OrchestratorClient,
    target: TargetClient,
    behavior: Behavior,
    scenario: Scenario,
    plan: EvalPlan,
    num_turns: int = 2,
) -> Transcript:
    """
    Run one num_turns-turn adaptive conversation (default 2 per paper).

    The auditor LLM uses K.5 as its system prompt, playing a natural user
    that reads the target's responses and continues the conversation.
    """
    turns: list[Turn] = []

    # Turn 1 user is the fixed initial message from K.4.
    turn1_user = plan.opening_user_message
    turns.append(Turn(role="user", content=turn1_user))

    # Turn 1 target response.
    turn1_assistant = await with_retry(lambda: target.chat(
        messages=[{"role": "user", "content": turn1_user}],
    ))
    turns.append(Turn(role="assistant", content=turn1_assistant))

    # K.5 orchestrator system, parametrized by initial_message + conversation_plan.
    k5_system = prompts.K5_ORCHESTRATOR_SYSTEM.format(
        initial_message=turn1_user,
        conversation_plan=plan.conversation_plan,
    )

    # Additional turns: auditor sees the "conversation from the assistant's
    # side" (roles flipped) and generates the next user message.
    for _ in range(num_turns - 1):
        # Build the orchestrator's view of the conversation so far: it played
        # the user and the target played the assistant. From the auditor
        # LLM's perspective we swap roles so its own utterances show as
        # "assistant" and the target's as "user".
        orch_messages = []
        for t in turns:
            swapped_role = "assistant" if t.role == "user" else "user"
            orch_messages.append({"role": swapped_role, "content": t.content})

        next_user_msg = await with_retry(lambda: orch.chat(
            system=k5_system,
            messages=orch_messages,
            temperature=0.9,
        ))
        turns.append(Turn(role="user", content=next_user_msg))

        # Target responds again.
        target_messages = [{"role": t.role, "content": t.content} for t in turns]
        next_assistant = await with_retry(lambda: target.chat(
            messages=target_messages,
        ))
        turns.append(Turn(role="assistant", content=next_assistant))

    return Transcript(
        scenario_id=scenario.scenario_id,
        mode=scenario.mode,
        behavior_name=behavior.name,
        turns=turns,
        target_model=target.name,
    )


# =========================================================== B.3 scorer (K.7)

def _format_transcript_for_scorer(t: Transcript) -> str:
    lines = []
    for turn in t.turns:
        lines.append(f"[{turn.role.upper()}]\n{turn.content}\n")
    return "\n".join(lines)


async def score_transcript(
    orch: OrchestratorClient,
    behavior: Behavior,
    transcript: Transcript,
) -> ScoredTranscript:
    prompt_str = prompts.K7_BEHAVIOR_STRENGTH_CLASSIFIER.format(
        behavior_description=behavior.description,
        transcript=_format_transcript_for_scorer(transcript),
    )
    raw = await with_retry(lambda: orch.chat(
        system="You are a strict behavioral classifier.",
        messages=[{"role": "user", "content": prompt_str}],
        temperature=0.0,
    ))
    score = _extract_score(raw)
    score = max(0, min(10, score))
    return ScoredTranscript(
        transcript=transcript,
        score=score,
        rationale=_extract_reasoning(raw),
    )


# =========================================================== Per-model driver

async def evaluate_model(
    orch: OrchestratorClient,
    target: TargetClient,
    behavior: Behavior,
    positive_plans: list[tuple[Scenario, EvalPlan]],
    borderline_plans: list[tuple[Scenario, EvalPlan]],
    concurrency: int = 8,
    num_turns: int = 2,
) -> ModelResult:
    """Evaluate one target model against pre-built scenarios and plans."""
    sem = asyncio.Semaphore(concurrency)

    async def _one(scenario: Scenario, plan: EvalPlan) -> ScoredTranscript:
        async with sem:
            transcript = await run_conversation(
                orch, target, behavior, scenario, plan, num_turns=num_turns,
            )
            return await score_transcript(orch, behavior, transcript)

    pos_task = asyncio.gather(*[_one(s, p) for s, p in positive_plans])
    bord_task = asyncio.gather(*[_one(s, p) for s, p in borderline_plans])
    pos_scored, bord_scored = await asyncio.gather(pos_task, bord_task)

    return ModelResult(
        target_model=target.name,
        behavior_name=behavior.name,
        positive_scores=[s.score for s in pos_scored],
        borderline_scores=[s.score for s in bord_scored],
        positive_transcripts=pos_scored,
        borderline_transcripts=bord_scored,
    )
