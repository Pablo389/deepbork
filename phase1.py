from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODAL_APP = Path(__file__).with_name("modal_phase1_app.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TritonBench-T Phase 1 on Modal for a predictions.jsonl file."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("outputs/predictions.jsonl"),
        help="Local predictions.jsonl produced by main.py.",
    )
    parser.add_argument(
        "--output-subdir",
        default="results/phase1",
        help="Volume-relative directory for Phase 1 artifacts.",
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


def run_phase1(
    predictions: Path,
    output_subdir: str,
    modal_app: Path = DEFAULT_MODAL_APP,
    modal_bin: str = "modal",
    capture_output: bool = False,
) -> subprocess.CompletedProcess:
    if not predictions.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions}")
    if not modal_app.exists():
        raise FileNotFoundError(f"Modal app file not found: {modal_app}")

    command = [
        modal_bin,
        "run",
        f"{modal_app}::evaluate_phase1_only",
        "--predictions",
        str(predictions),
        "--output-subdir",
        output_subdir,
    ]
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=capture_output,
    )


def extract_phase1_summary(output: str) -> dict:
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
        raise ValueError("could not find Phase 1 JSON summary in Modal output")
    return summary


def evaluate_phase1_local(
    predictions: Path,
    output_subdir: str,
    modal_app: Path = DEFAULT_MODAL_APP,
    modal_bin: str = "modal",
) -> dict:
    try:
        result = run_phase1(
            predictions=predictions,
            output_subdir=output_subdir,
            modal_app=modal_app,
            modal_bin=modal_bin,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            sys.stdout.write(exc.stdout)
        if exc.stderr:
            sys.stderr.write(exc.stderr)
        raise
    return extract_phase1_summary((result.stdout or "") + "\n" + (result.stderr or ""))


def main() -> None:
    args = parse_args()
    try:
        run_phase1(
            predictions=args.predictions,
            output_subdir=args.output_subdir,
            modal_app=args.modal_app,
            modal_bin=args.modal_bin,
        )
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
