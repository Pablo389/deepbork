from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    input_artifact: str
    output_artifact: str | None


@dataclass
class EvalContext:
    output_subdir: str
    predictions: Path | None = None
    call_acc_dir: str | None = None
    perf_results_dir: str | None = None


@dataclass
class PhaseResult:
    phase: str
    attempted_files: list[str]
    passed_files: list[str]
    failed_files: list[str]
    metrics: dict
    artifacts: dict[str, str]
    stdout_tail: str
    stderr_tail: str
    failed: bool


PHASE_ORDER = ("phase1", "phase2", "phase3")

PHASES = {
    "phase1": PhaseSpec(
        name="phase1",
        input_artifact="predictions_path",
        output_artifact="call_acc_dir",
    ),
    "phase2": PhaseSpec(
        name="phase2",
        input_artifact="call_acc_dir",
        output_artifact="call_acc_dir",
    ),
    "phase3": PhaseSpec(
        name="phase3",
        input_artifact="call_acc_dir",
        output_artifact="perf_results_dir",
    ),
}
