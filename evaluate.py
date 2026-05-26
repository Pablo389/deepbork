from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from evaluation.model import PHASE_ORDER, PHASES


DEFAULT_MODAL_APP = Path(__file__).with_name("modal_eval_app.py")
DEFAULT_MODAL_BIN = "modal"
EVAL_JSON_START = "DEEPBORK_EVAL_JSON_START"
EVAL_JSON_END = "DEEPBORK_EVAL_JSON_END"
DEFAULT_PHASE1_OUTPUT_SUBDIR = "results/phase1"
DEFAULT_PHASE2_OUTPUT_SUBDIR = "results/phase2"
DEFAULT_PHASE3_OUTPUT_SUBDIR = "results/phase3"
DEFAULT_ALL_OUTPUT_SUBDIR = "results/all"
DEFAULT_OUTPUT_SUBDIRS = {
    "phase1": DEFAULT_PHASE1_OUTPUT_SUBDIR,
    "phase2": DEFAULT_PHASE2_OUTPUT_SUBDIR,
    "phase3": DEFAULT_PHASE3_OUTPUT_SUBDIR,
    "all": DEFAULT_ALL_OUTPUT_SUBDIR,
}
STAGE_ORDER = PHASE_ORDER
VALID_MODES = (*PHASE_ORDER, "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TritonBench-T evaluation phases on Modal."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/predictions.jsonl"),
        help="Local predictions.jsonl produced by main.py.",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="Volume-relative directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="phase1",
        help="Run one evaluation phase or the full Phase 1+2+3 pipeline.",
    )
    parser.add_argument(
        "--call-acc-subdir",
        default=None,
        help="Volume-relative call_acc folder for --mode phase2 or --mode phase3.",
    )
    return parser.parse_args()


def default_output_subdir_for_mode(mode: str) -> str:
    validate_mode(mode)
    return DEFAULT_OUTPUT_SUBDIRS[mode]


def validate_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"unknown evaluation mode: {mode}")


def extract_evaluation_summary(output: str) -> dict:
    start_index = output.rfind(EVAL_JSON_START)
    end_index = output.rfind(EVAL_JSON_END)
    if start_index != -1 or end_index != -1:
        if start_index == -1 or end_index == -1 or end_index < start_index:
            raise ValueError("found incomplete evaluation JSON sentinel block")
        payload_start = start_index + len(EVAL_JSON_START)
        payload = output[payload_start:end_index].strip()
        try:
            summary = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("could not parse evaluation JSON sentinel block") from exc
        if not isinstance(summary, dict):
            raise ValueError("evaluation JSON sentinel block did not contain an object")
        return summary

    decoder = json.JSONDecoder()
    pipeline_summary = None
    phase_summary = None
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if "results" in value and "artifacts" in value:
            pipeline_summary = value
        elif "phase" in value and "metrics" in value:
            phase_summary = value
    summary = pipeline_summary or phase_summary
    if summary is None:
        raise ValueError("could not find evaluation JSON summary in Modal output")
    return summary


def echo_process_output(result: subprocess.CompletedProcess | subprocess.CalledProcessError) -> None:
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)


def modal_json(command: list[str], echo: bool = False) -> dict:
    if not DEFAULT_MODAL_APP.exists():
        raise FileNotFoundError(f"Modal app file not found: {DEFAULT_MODAL_APP}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        echo_process_output(exc)
        raise
    if echo:
        echo_process_output(result)
    return extract_evaluation_summary((result.stdout or "") + "\n" + (result.stderr or ""))


def run_modal_phase(
    phase: str,
    output_subdir: str,
    predictions: Path | None = None,
    call_acc_subdir: str | None = None,
    echo: bool = False,
) -> dict:
    if phase not in PHASES:
        raise ValueError(f"unknown evaluation phase: {phase}")
    if phase == "phase1":
        if predictions is None:
            raise ValueError("phase1 requires predictions")
        if not predictions.exists():
            raise FileNotFoundError(f"predictions file not found: {predictions}")
    elif not call_acc_subdir:
        raise ValueError(f"{phase} requires --call-acc-subdir")

    command = [
        DEFAULT_MODAL_BIN,
        "run",
        f"{DEFAULT_MODAL_APP}::run_phase",
        "--phase",
        phase,
        "--output-subdir",
        output_subdir,
    ]
    if phase == "phase1":
        command.extend(["--predictions", str(predictions)])
    else:
        command.extend(["--call-acc-subdir", call_acc_subdir or ""])
    return modal_json(command, echo=echo)


def run_modal_pipeline(
    predictions: Path,
    output_subdir: str,
    echo: bool = False,
) -> dict:
    if not predictions.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions}")
    command = [
        DEFAULT_MODAL_BIN,
        "run",
        f"{DEFAULT_MODAL_APP}::run_pipeline",
        "--predictions",
        str(predictions),
        "--output-subdir",
        output_subdir,
    ]
    return modal_json(command, echo=echo)


def failed_phase(results: list[dict]) -> str | None:
    for result in results:
        if result.get("failed"):
            return result.get("phase")
    return None


def passed_through_phase(results: list[dict]) -> str | None:
    last_passed = None
    for result in results:
        if result.get("failed"):
            return last_passed
        last_passed = result.get("phase")
    return last_passed


def evaluation_summary(
    mode: str,
    results: list[dict],
    artifacts: dict[str, str] | None = None,
) -> dict:
    if not results:
        raise ValueError("evaluation summary requires at least one phase result")
    top_artifacts = artifacts or dict(results[-1].get("artifacts", {}))
    return {
        "mode": mode,
        "failed_phase": failed_phase(results),
        "passed_through": passed_through_phase(results),
        "artifacts": top_artifacts,
        "results": results,
    }


def evaluate_phase(
    phase: str,
    output_subdir: str,
    predictions: Path | None = None,
    call_acc_subdir: str | None = None,
    echo: bool = False,
) -> dict:
    result = run_modal_phase(
        phase=phase,
        predictions=predictions,
        call_acc_subdir=call_acc_subdir,
        output_subdir=output_subdir,
        echo=echo,
    )
    return evaluation_summary(phase, [result])


def evaluate_all(
    predictions: Path,
    output_subdir: str,
    echo: bool = False,
) -> dict:
    pipeline = run_modal_pipeline(predictions=predictions, output_subdir=output_subdir, echo=echo)
    return evaluation_summary(
        "all",
        list(pipeline.get("results", [])),
        artifacts=dict(pipeline.get("artifacts", {})),
    )


def evaluate_local(
    mode: str,
    output_subdir: str,
    predictions: Path | None = None,
    call_acc_subdir: str | None = None,
    echo: bool = False,
) -> dict:
    validate_mode(mode)
    if mode == "all":
        if predictions is None:
            raise ValueError("all requires predictions")
        return evaluate_all(predictions=predictions, output_subdir=output_subdir, echo=echo)
    return evaluate_phase(
        phase=mode,
        predictions=predictions,
        call_acc_subdir=call_acc_subdir,
        output_subdir=output_subdir,
        echo=echo,
    )


def phase_result(summary: dict, phase: str) -> dict | None:
    for result in summary.get("results", []):
        if result.get("phase") == phase:
            return result
    return None


def is_passed(summary: dict, op_file: str, target_stage: str) -> bool:
    if target_stage not in PHASES:
        raise ValueError(f"unknown target stage: {target_stage}")
    result = phase_result(summary, target_stage)
    return op_file in set((result or {}).get("passed_files", []))


def passed_through(summary: dict, op_file: str) -> str | None:
    for stage in reversed(PHASE_ORDER):
        if is_passed(summary, op_file, stage):
            return stage
    return None


def failed_stage(summary: dict, target_stage: str) -> str:
    if target_stage not in PHASES:
        raise ValueError(f"unknown target stage: {target_stage}")
    if summary.get("failed_phase"):
        return summary["failed_phase"]
    for stage in PHASE_ORDER[: PHASE_ORDER.index(target_stage) + 1]:
        result = phase_result(summary, stage)
        if result is not None and result.get("failed"):
            return stage
    return target_stage


def phase_line(summary: dict, stage: str) -> str:
    result = phase_result(summary, stage)
    if result is None:
        return f"{stage}: not-run"
    metrics = result.get("metrics", {})
    if stage == "phase3":
        status = metrics.get("status", "not-run")
        speedup = metrics.get("speedup_vs_pytorch")
        return f"phase3: {status}, speedup={speedup}"
    passed = metrics.get("passed", 0)
    failed = metrics.get("failed", 0)
    rate = metrics.get("rate", 0)
    return f"{stage}: {passed} passed, {failed} failed ({rate}%)"


def main() -> None:
    args = parse_args()
    output_subdir = args.output_subdir or default_output_subdir_for_mode(args.mode)
    call_acc_subdir = args.call_acc_subdir

    try:
        summary = evaluate_local(
            mode=args.mode,
            predictions=args.predictions,
            call_acc_subdir=call_acc_subdir,
            output_subdir=output_subdir,
            echo=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
