"""Modal Phase 1 evaluator for deepbork predictions.

This app accepts a local ``predictions.jsonl`` produced by ``main.py``, uploads
it to a Modal Volume, and runs only TritonBench-T Phase 1
(``0_call_acc.py::call_4file``). It intentionally stops before execution
accuracy and performance benchmarking so it can become the inner check in the
future agentic repair loop.

Usage:
    modal run modal_phase1_app.py::evaluate_phase1_only \
        --predictions outputs/predictions.jsonl
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import modal

from tritonbench_helpers import DEFAULT_METADATA_FILE, load_metadata, prediction_files


APP_NAME = "deepbork-phase1"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

# T4 is enough for TritonBench-T call accuracy and keeps smoke tests cheap.
DEFAULT_GPU = os.environ.get("DEEPBORK_PHASE1_GPU", "T4")

VOLUME_NAME = os.environ.get("DEEPBORK_MODAL_VOLUME", "deepbork-phase1-data")
DATA_DIR = "/data"
REPO_DIR = "/opt/TritonBench"


PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{REPO_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/0_call_acc.py"""
)


image = (
    modal.Image.from_registry(
        # TritonBench eval scripts use Python >=3.12 f-string syntax.
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "triton==3.1.0",
        "tqdm==4.66.5",
        "numpy<2",
    )
    .run_commands(f"git clone --depth 1 {TRITONBENCH_REPO} {REPO_DIR}")
    .run_commands(PATCH_CALL_ACC)
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py"
    )
    .add_local_python_source("tritonbench_helpers")
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _upload_local_predictions(local_path: Path) -> str:
    """Upload a local predictions.jsonl to the Modal Volume."""
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    remote = f"uploads/{local_path.name}"
    print(f"uploading {local_path} -> volume://{remote}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote)
    return remote


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 3,
    volumes={DATA_DIR: data_volume},
)
def evaluate_phase1(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results/phase1",
) -> dict:
    """Run TritonBench-T Phase 1 against an existing predictions.jsonl."""
    pred_full = Path(DATA_DIR) / predictions_path
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")

    out_dir = Path(DATA_DIR) / output_subdir
    call_acc_dir = out_dir / "call_acc"
    logs_dir = out_dir / "logs"
    if call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")

    import call_acc  # noqa: E402

    metadata = load_metadata(Path(REPO_DIR) / "data" / DEFAULT_METADATA_FILE)
    attempted = prediction_files(pred_full, metadata)
    total = len(attempted)

    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(_Tee(sys.stdout, stdout_buffer)):
        with redirect_stderr(_Tee(sys.stderr, stderr_buffer)):
            call_acc.call_4file(str(pred_full), str(call_acc_dir), gpus=[0])

    stdout_text = stdout_buffer.getvalue()
    stderr_text = stderr_buffer.getvalue()
    (logs_dir / "phase1_stdout.txt").write_text(stdout_text)
    (logs_dir / "phase1_stderr.txt").write_text(stderr_text)

    passed = sorted(path.name for path in call_acc_dir.glob("*.py"))
    passed_set = set(passed)
    failed = [file_name for file_name in attempted if file_name not in passed_set]

    data_volume.commit()

    return {
        "total_predictions": total,
        "phase1_call_acc": {
            "passed": len(passed),
            "failed": len(failed),
            "rate": round(100 * len(passed) / total, 2) if total else 0,
        },
        "attempted_files": attempted,
        "passed_files": passed,
        "failed_files": failed,
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
        "call_acc_dir": f"{output_subdir}/call_acc",
        "logs_dir": f"{output_subdir}/logs",
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
    }


@app.local_entrypoint()
def evaluate_phase1_only(
    predictions: str,
    output_subdir: str = "results/phase1",
):
    """Upload a local predictions.jsonl and run only Phase 1 on Modal."""
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate_phase1.remote(
        predictions_path=remote,
        output_subdir=output_subdir,
    )
    print(json.dumps(summary, indent=2))
