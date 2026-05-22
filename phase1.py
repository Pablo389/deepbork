from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODAL_APP = Path(__file__).with_name("modal_phase1_app.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TritonBench-T Phase 1 or Phase 1+2 on Modal for a predictions.jsonl file."
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
        help="Volume-relative directory for evaluation artifacts. Defaults to results/phase1 or results/phase1_phase2.",
    )
    parser.add_argument(
        "--target-phase",
        choices=["phase1", "phase2", "all"],
        default="phase1",
        help="Run Phase 1 only, Phase 2 from an existing call_acc folder, or Phase 1 followed by Phase 2.",
    )
    parser.add_argument(
        "--call-acc-subdir",
        default="",
        help="Volume-relative call_acc folder for --target-phase phase2, e.g. results/phase1/call_acc.",
    )
    parser.add_argument(
        "--modal-app",
        type=Path,
        default=DEFAULT_MODAL_APP,
        help="Path to the Modal app file.",
    )
    parser.add_argument(
        "--modal-bin",
        default="modal",
        help="Modal executable to invoke.",
    )
    return parser.parse_args()


def run_evaluation(
    predictions: Path,
    output_subdir: str,
    modal_app: Path = DEFAULT_MODAL_APP,
    modal_bin: str = "modal",
    capture_output: bool = False,
    target_phase: str = "phase1",
    call_acc_subdir: str = "",
) -> subprocess.CompletedProcess:
    if target_phase != "phase2" and not predictions.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions}")
    if not modal_app.exists():
        raise FileNotFoundError(f"Modal app file not found: {modal_app}")

    entrypoint = "evaluate_phase1_only"
    if target_phase == "phase2":
        if not call_acc_subdir:
            raise ValueError("--call-acc-subdir is required for --target-phase phase2")
        entrypoint = "evaluate_phase2_only"
    elif target_phase == "all":
        entrypoint = "evaluate_all_entrypoint"
    elif target_phase != "phase1":
        raise ValueError(f"unknown target phase: {target_phase}")

    command = [
        modal_bin,
        "run",
        f"{modal_app}::{entrypoint}",
    ]
    if target_phase == "phase2":
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


def run_phase1(
    predictions: Path,
    output_subdir: str,
    modal_app: Path = DEFAULT_MODAL_APP,
    modal_bin: str = "modal",
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    return run_evaluation(
        predictions=predictions,
        output_subdir=output_subdir,
        modal_app=modal_app,
        modal_bin=modal_bin,
        capture_output=capture_output,
        target_phase="phase1",
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
        if isinstance(value, dict) and "phase1_call_acc" in value:
            summary = value
    if summary is None:
        raise ValueError("could not find evaluation JSON summary in Modal output")
    return summary


def extract_phase1_summary(output: str) -> dict:
    return extract_evaluation_summary(output)


def evaluate_local(
    predictions: Path,
    output_subdir: str,
    modal_app: Path = DEFAULT_MODAL_APP,
    modal_bin: str = "modal",
    target_phase: str = "phase1",
    call_acc_subdir: str = "",
) -> dict:
    try:
        result = run_evaluation(
            predictions=predictions,
            output_subdir=output_subdir,
            modal_app=modal_app,
            modal_bin=modal_bin,
            capture_output=True,
            target_phase=target_phase,
            call_acc_subdir=call_acc_subdir,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
    return extract_evaluation_summary((result.stdout or "") + "\n" + (result.stderr or ""))


def evaluate_phase1_local(
    predictions: Path,
    output_subdir: str,
    modal_app: Path = DEFAULT_MODAL_APP,
    modal_bin: str = "modal",
) -> dict:
    return evaluate_local(
        predictions=predictions,
        output_subdir=output_subdir,
        modal_app=modal_app,
        modal_bin=modal_bin,
        target_phase="phase1",
    )


def main() -> None:
    args = parse_args()
    output_subdir = args.output_subdir or (
        "results/phase2"
        if args.target_phase == "phase2"
        else "results/phase1_phase2"
        if args.target_phase == "all"
        else "results/phase1"
    )
    try:
        run_evaluation(
            predictions=args.predictions,
            output_subdir=output_subdir,
            modal_app=args.modal_app,
            modal_bin=args.modal_bin,
            target_phase=args.target_phase,
            call_acc_subdir=args.call_acc_subdir,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
