"""AuditBench-lite: a reconstruction of the B.1+B.3 pipeline from
Sheshadri et al. 2026 (arXiv:2602.22755) for measuring hidden-behavior
strength in target models, extended for subliminal-transfer experiments.
"""

from .models import Behavior, Scenario, EvalPlan, Transcript, ScoredTranscript, ModelResult
from .clients import OrchestratorClient, OpenAIOrchestratorClient, AnthropicTargetClient, OpenAICompatTargetClient
from .runner import run_transfer_experiment, TransferExperimentResult, generate_evaluation_plans, PlanGenerationResult, run_evaluation_from_plans, load_plans_file

__all__ = [
    "Behavior",
    "Scenario",
    "EvalPlan",
    "Transcript",
    "ScoredTranscript",
    "ModelResult",
    "OrchestratorClient",
    "OpenAIOrchestratorClient",
    "AnthropicTargetClient",
    "OpenAICompatTargetClient",
    "run_transfer_experiment",
    "TransferExperimentResult",
    "generate_evaluation_plans",
    "PlanGenerationResult",
    "run_evaluation_from_plans",
    "load_plans_file",
]
