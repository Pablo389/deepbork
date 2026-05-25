from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODAL_APP = Path(__file__).with_name("modal_eval_app.py")
DEFAULT_MODAL_BIN = "modal"
DEFAULT_PHASE1_OUTPUT_SUBDIR = "results/phase1"
DEFAULT_PHASE2_OUTPUT_SUBDIR = "results/phase2"
DEFAULT_PHASE3_OUTPUT_SUBDIR = "results/phase3"
DEFAULT_PHASE1_PHASE2_OUTPUT_SUBDIR = "results/phase1_phase2"
DEFAULT_ALL_OUTPUT_SUBDIR = "results/all"
DEFAULT_CALL_ACC_SUBDIR = f"{DEFAULT_PHASE1_OUTPUT_SUBDIR}/call_acc"
DEFAULT_PHASE3_CALL_ACC_SUBDIR = f"{DEFAULT_PHASE2_OUTPUT_SUBDIR}/call_acc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone or sequential TritonBench-T evaluation stages on Modal."
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
        choices=["phase1", "phase2", "phase3", "all"],
        default="phase1",
        help="Run a standalone evaluation phase or the full Phase 1+2+3 pipeline.",
    )
    parser.add_argument(
        "--call-acc-subdir",
        default=DEFAULT_CALL_ACC_SUBDIR,
        help="Volume-relative call_acc folder for --mode phase2 or --mode phase3.",
    )
    return parser.parse_args()


def run_evaluation(
    predictions: Path,
    output_subdir: str,
    capture_output: bool = False,
    mode: str = "phase1",
    call_acc_subdir: str = "",
) -> subprocess.CompletedProcess:
    if mode not in {"phase2", "phase3"} and not predictions.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions}")
    if not DEFAULT_MODAL_APP.exists():
        raise FileNotFoundError(f"Modal app file not found: {DEFAULT_MODAL_APP}")

    entrypoint = "evaluate_phase1_only"
    if mode == "phase2":
        entrypoint = "evaluate_phase2_only"
    elif mode == "phase3":
        entrypoint = "evaluate_phase3_only"
    elif mode == "phase1_phase2":
        entrypoint = "evaluate_through_phase2_entrypoint"
    elif mode == "all":
        entrypoint = "evaluate_all_entrypoint"
    elif mode != "phase1":
        raise ValueError(f"unknown evaluation mode: {mode}")

    command = [
        DEFAULT_MODAL_BIN,
        "run",
        f"{DEFAULT_MODAL_APP}::{entrypoint}",
    ]
    if mode in {"phase2", "phase3"}:
        command.extend(["--call-acc-subdir", call_acc_subdir])
    else:
        command.extend(["--predictions", str(predictions)])
    command.extend(["--output-subdir", output_subdir])
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )


def extract_evaluation_summary(output: str) -> dict:
    decoder = json.JSONDecoder()
    summary = None
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (
            "phase1_call_acc" in value
            or "phase2_exec_acc" in value
            or "phase3_efficiency" in value
        ):
            summary = value
    if summary is None:
        raise ValueError("could not find evaluation JSON summary in Modal output")
    return summary


def evaluate_local(
    predictions: Path,
    output_subdir: str,
    mode: str = "phase1",
    call_acc_subdir: str = "",
) -> dict:
    try:
        result = run_evaluation(
            predictions=predictions,
            output_subdir=output_subdir,
            capture_output=True,
            mode=mode,
            call_acc_subdir=call_acc_subdir,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
    return extract_evaluation_summary((result.stdout or "") + "\n" + (result.stderr or ""))


def evaluate_through(
    predictions: Path,
    output_subdir: str,
    target_stage: str,
) -> dict:
    if target_stage == "phase1":
        return evaluate_local(predictions=predictions, output_subdir=output_subdir, mode="phase1")
    if target_stage == "phase2":
        return evaluate_local(
            predictions=predictions,
            output_subdir=output_subdir,
            mode="phase1_phase2",
        )
    if target_stage == "phase3":
        return evaluate_local(predictions=predictions, output_subdir=output_subdir, mode="all")
    raise ValueError(f"unknown target stage: {target_stage}")


def main() -> None:
    args = parse_args()
    output_subdir = args.output_subdir or (
        DEFAULT_PHASE2_OUTPUT_SUBDIR
        if args.mode == "phase2"
        else DEFAULT_PHASE3_OUTPUT_SUBDIR
        if args.mode == "phase3"
        else DEFAULT_ALL_OUTPUT_SUBDIR
        if args.mode == "all"
        else DEFAULT_PHASE1_OUTPUT_SUBDIR
    )
    call_acc_subdir = DEFAULT_PHASE3_CALL_ACC_SUBDIR if args.mode == "phase3" else args.call_acc_subdir
    try:
        run_evaluation(
            predictions=args.predictions,
            output_subdir=output_subdir,
            mode=args.mode,
            call_acc_subdir=call_acc_subdir,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
