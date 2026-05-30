"""Modal evaluator for deepbork predictions.

This app is the Modal execution adapter for TritonBench-T evaluation phases:

- Phase 1: upload/use a predictions.jsonl and run call accuracy.
- Phase 2: use an existing Modal-volume call_acc folder and run execution accuracy.
- Phase 3: use an existing Modal-volume call_acc folder and run efficiency benchmarking.
- All: run Phase 1, Phase 2, then Phase 3 in one Modal function call.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

from constants import (
    DEFAULT_MODAL_APP_NAME,
    DEFAULT_MODAL_VOLUME,
    EVAL_JSON_END,
    EVAL_JSON_START,
)
from evaluation.model import PHASE_ORDER, PHASES
from tritonbench_helpers import DEFAULT_METADATA_FILE, load_metadata, prediction_files


APP_NAME = DEFAULT_MODAL_APP_NAME
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

# Deployment knobs live in code so Modal captures them when the app is deployed.
# Change these constants before `modal deploy modal_eval_app.py` to adjust
# compute, artifact storage, or warm-container behavior.
#
# Modal references:
# - GPU selection: https://modal.com/docs/guide/gpu
# - Scaling and `scaledown_window`: https://modal.com/docs/guide/scale
# - Cold starts and warm containers: https://modal.com/docs/guide/cold-start
DEFAULT_GPU = "A100-40GB"
VOLUME_NAME = DEFAULT_MODAL_VOLUME
SCALEDOWN_WINDOW = 180
DATA_DIR = "/data"
REPO_DIR = "/opt/TritonBench"


PATCH_CALL_ACC = (
    f"""sed -i """
    f"""-e 's|^statis_path = .*|statis_path = "{REPO_DIR}/data/TritonBench_T_v1.jsonl"|' """
    f"""-e 's|^py_folder = .*|py_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/0_call_acc.py"""
)

PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/1_exe_acc.py"""
)

PATCH_PERF = (
    f"""sed -i 's|^gpu_count = .*|gpu_count = 1|' """
    f"""{REPO_DIR}/performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py"""
)


image = (
    modal.Image.from_registry(
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
    .run_commands(PATCH_CALL_ACC, PATCH_EXE_ACC, PATCH_PERF)
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py",
        f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py",
    )
    .add_local_python_source("tritonbench_helpers")
    .add_local_python_source("evaluation")
    .add_local_python_source("constants")
)

app = modal.App(APP_NAME, image=image)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _volume_path(volume_relative_path: str) -> Path:
    return Path(DATA_DIR) / volume_relative_path


def _eval_dir() -> str:
    eval_dir = f"{REPO_DIR}/EVAL/eval_T"
    if eval_dir not in sys.path:
        sys.path.insert(0, eval_dir)
    os.environ["PYTHONPATH"] = eval_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    return eval_dir


def _prepare_output(output_subdir: str, reset_call_acc: bool = True) -> tuple[Path, Path, Path]:
    out_dir = _volume_path(output_subdir)
    call_acc_dir = out_dir / "call_acc"
    logs_dir = out_dir / "logs"
    if reset_call_acc and call_acc_dir.exists():
        shutil.rmtree(call_acc_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, call_acc_dir, logs_dir


def _run_python(command_code: str, args: list[str], phase_name: str) -> tuple[str, str]:
    eval_dir = _eval_dir()
    env = os.environ.copy()
    env["PYTHONPATH"] = eval_dir + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", command_code, *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    sys.stderr.write(proc.stderr)
    sys.stderr.flush()
    if proc.returncode:
        raise RuntimeError(f"{phase_name} failed with exit code {proc.returncode}")
    return proc.stdout, proc.stderr


def _write_phase_logs(logs_dir: Path, phase_name: str, stdout_text: str, stderr_text: str) -> None:
    (logs_dir / f"{phase_name}_stdout.txt").write_text(stdout_text)
    (logs_dir / f"{phase_name}_stderr.txt").write_text(stderr_text)


def _run_subprocess(command: list[str], cwd: str, phase_name: str) -> tuple[str, str, int]:
    proc = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    sys.stderr.write(proc.stderr)
    sys.stderr.flush()
    if proc.returncode:
        message = f"{phase_name} failed with exit code {proc.returncode}\n"
        print(message, flush=True)
        return proc.stdout, proc.stderr + message, proc.returncode
    return proc.stdout, proc.stderr, proc.returncode


def _run_call_acc(predictions_path: Path, call_acc_dir: Path) -> tuple[str, str]:
    return _run_python(
        command_code=(
            "import call_acc, sys; "
            "call_acc.call_4file(sys.argv[1], sys.argv[2], gpus=[0])"
        ),
        args=[str(predictions_path), str(call_acc_dir)],
        phase_name="Phase 1 call_acc",
    )


def _run_exe_acc(call_acc_dir: Path) -> tuple[str, str]:
    return _run_python(
        command_code="import exe_acc, sys; exe_acc.execute_4folder(sys.argv[1], gpus=[0])",
        args=[str(call_acc_dir)],
        phase_name="Phase 2 exe_acc",
    )


def _prediction_files(predictions_path: Path) -> list[str]:
    metadata = load_metadata(Path(REPO_DIR) / "data" / DEFAULT_METADATA_FILE)
    return prediction_files(predictions_path, metadata)


def _artifacts(output_subdir: str, include_perf_results: bool = False) -> dict[str, str]:
    artifacts = {
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
        "call_acc_dir": f"{output_subdir}/call_acc",
        "logs_dir": f"{output_subdir}/logs",
    }
    if include_perf_results:
        artifacts["perf_results_dir"] = f"{output_subdir}/perf_results"
    return artifacts


def _phase_summary(
    phase: str,
    output_subdir: str,
    attempted: list[str],
    passed: list[str],
    failed: list[str],
    metrics: dict,
    stdout_text: str,
    stderr_text: str,
) -> dict:
    status = metrics.get("status")
    return {
        "phase": phase,
        "attempted_files": attempted,
        "passed_files": passed,
        "failed_files": failed,
        "metrics": metrics,
        "artifacts": _artifacts(output_subdir, include_perf_results=phase == "phase3"),
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
        "failed": bool(failed) or status == "failed",
    }


def _phase1_result(
    output_subdir: str,
    attempted: list[str],
    passed: list[str],
    failed: list[str],
    stdout_text: str,
    stderr_text: str,
) -> dict:
    total = len(attempted)
    return _phase_summary(
        phase="phase1",
        output_subdir=output_subdir,
        attempted=attempted,
        passed=passed,
        failed=failed,
        metrics={
            "passed": len(passed),
            "failed": len(failed),
            "rate": round(100 * len(passed) / total, 2) if total else 0,
        },
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def _phase2_result(
    output_subdir: str,
    candidate_files: list[str],
    passed: list[str],
    failed: list[str],
    stdout_text: str,
    stderr_text: str,
) -> dict:
    total = len(candidate_files)
    return _phase_summary(
        phase="phase2",
        output_subdir=output_subdir,
        attempted=candidate_files,
        passed=passed,
        failed=failed,
        metrics={
            "passed": len(passed),
            "failed": len(failed),
            "rate": round(100 * len(passed) / total, 2) if total else 0,
        },
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def _parse_speedup(stdout_text: str) -> float | None:
    for line in stdout_text.splitlines():
        if not line.startswith("speed up:"):
            continue
        try:
            return float(line.split(":", 1)[1].strip())
        except ValueError:
            return None
    return None


def _phase3_result(
    output_subdir: str,
    candidate_files: list[str],
    speedup: float | None,
    stdout_text: str,
    stderr_text: str,
    returncode: int,
) -> dict:
    status = "skipped" if not candidate_files else "success" if returncode == 0 else "failed"
    passed = candidate_files if returncode == 0 else []
    failed = [] if returncode == 0 else candidate_files
    return _phase_summary(
        phase="phase3",
        output_subdir=output_subdir,
        attempted=candidate_files,
        passed=passed,
        failed=failed,
        metrics={
            "status": status,
            "speedup_vs_pytorch": speedup,
            "returncode": returncode,
            "raw_output_tail": stdout_text[-4000:],
        },
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def _run_phase1(predictions_path: Path, output_subdir: str) -> tuple[dict, Path]:
    _, call_acc_dir, logs_dir = _prepare_output(output_subdir, reset_call_acc=True)
    attempted = _prediction_files(predictions_path)

    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    stdout_text, stderr_text = _run_call_acc(predictions_path, call_acc_dir)
    _write_phase_logs(logs_dir, "phase1", stdout_text, stderr_text)

    passed = sorted(path.name for path in call_acc_dir.glob("*.py"))
    passed_set = set(passed)
    failed = [file_name for file_name in attempted if file_name not in passed_set]
    return _phase1_result(output_subdir, attempted, passed, failed, stdout_text, stderr_text), call_acc_dir


def _run_phase2_on_call_acc(call_acc_dir: Path, output_subdir: str) -> dict:
    logs_dir = _volume_path(output_subdir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    candidate_files = sorted(path.name for path in call_acc_dir.glob("*.py"))

    print("\n" + "=" * 70 + "\n=== Phase 2: execution accuracy ===\n" + "=" * 70, flush=True)
    if candidate_files:
        stdout_text, stderr_text = _run_exe_acc(call_acc_dir)
    else:
        stdout_text = "skipped Phase 2: no Phase 1 survivors\n"
        stderr_text = ""
        print(stdout_text, flush=True)
    _write_phase_logs(logs_dir, "phase2", stdout_text, stderr_text)

    passed = sorted(path.name for path in call_acc_dir.glob("*.py"))
    passed_set = set(passed)
    failed = [file_name for file_name in candidate_files if file_name not in passed_set]
    return _phase2_result(output_subdir, candidate_files, passed, failed, stdout_text, stderr_text)


def _run_phase3_on_call_acc(call_acc_dir: Path, output_subdir: str) -> dict:
    output_dir = _volume_path(output_subdir)
    logs_dir = output_dir / "logs"
    perf_results_dir = output_dir / "perf_results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    if perf_results_dir.exists():
        shutil.rmtree(perf_results_dir)
    perf_results_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(path.name for path in call_acc_dir.glob("*.py"))
    print("\n" + "=" * 70 + "\n=== Phase 3: efficiency ===\n" + "=" * 70, flush=True)

    if not candidate_files:
        stdout_text = "skipped Phase 3: no Phase 2 survivors\n"
        stderr_text = ""
        print(stdout_text, flush=True)
        _write_phase_logs(logs_dir, "phase3", stdout_text, stderr_text)
        return _phase3_result(output_subdir, [], None, stdout_text, stderr_text, 0)

    perf_root = f"{REPO_DIR}/performance_metrics/perf_T"
    eval_root = f"{REPO_DIR}/EVAL/eval_T"
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    returncode = 0

    commands = [
        (
            [
                sys.executable,
                "run_bench/write_file.py",
                "--input_folder_path",
                str(call_acc_dir),
                "--results_path",
                str(perf_results_dir),
            ],
            perf_root,
            "Phase 3 write_file",
        ),
        (
            [sys.executable, "run_bench/multiprocess_gpu_run.py"],
            perf_root,
            "Phase 3 benchmark",
        ),
        (
            [
                sys.executable,
                "2_efficiency.py",
                "--gen_folder",
                str(perf_results_dir),
            ],
            eval_root,
            "Phase 3 efficiency",
        ),
    ]

    for command, cwd, phase_name in commands:
        stdout, stderr, code = _run_subprocess(command, cwd, phase_name)
        stdout_parts.append(stdout)
        stderr_parts.append(stderr)
        if code:
            returncode = code
            break

    stdout_text = "".join(stdout_parts)
    stderr_text = "".join(stderr_parts)
    speedup = _parse_speedup(stdout_text) if returncode == 0 else None
    _write_phase_logs(logs_dir, "phase3", stdout_text, stderr_text)
    return _phase3_result(output_subdir, candidate_files, speedup, stdout_text, stderr_text, returncode)


def _copy_call_acc(call_acc_subdir: str, output_subdir: str) -> Path:
    source = _volume_path(call_acc_subdir)
    if not source.exists():
        raise FileNotFoundError(f"call_acc folder not found in volume: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"call_acc path is not a folder: {source}")

    destination = _volume_path(output_subdir) / "call_acc"
    if source.resolve() == destination.resolve():
        _prepare_output(output_subdir, reset_call_acc=False)
        return destination

    _, destination, _ = _prepare_output(output_subdir, reset_call_acc=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination


def _upload_local_predictions(local_path: Path) -> str:
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
def run_phase_remote(request: dict) -> dict:
    phase = request.get("phase")
    output_subdir = request.get("output_subdir", "")
    predictions_path = request.get("predictions_path", "")
    call_acc_subdir = request.get("call_acc_subdir", "")

    if phase not in PHASES:
        raise ValueError(f"unknown evaluation phase: {phase}")
    if not output_subdir:
        raise ValueError("output_subdir is required")

    if phase == "phase1":
        if not predictions_path:
            raise ValueError("phase1 requires predictions_path")
        pred_full = _volume_path(predictions_path)
        if not pred_full.exists():
            raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")
        result, _ = _run_phase1(pred_full, output_subdir)
    elif phase == "phase2":
        if not call_acc_subdir:
            raise ValueError("phase2 requires call_acc_subdir")
        call_acc_dir = _copy_call_acc(call_acc_subdir, output_subdir)
        result = _run_phase2_on_call_acc(call_acc_dir, output_subdir)
    elif phase == "phase3":
        if not call_acc_subdir:
            raise ValueError("phase3 requires call_acc_subdir")
        call_acc_dir = _copy_call_acc(call_acc_subdir, output_subdir)
        result = _run_phase3_on_call_acc(call_acc_dir, output_subdir)

    data_volume.commit()
    return result


def _pipeline_artifacts(output_subdir: str) -> dict[str, str]:
    return _artifacts(output_subdir, include_perf_results=True)


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 3,
    volumes={DATA_DIR: data_volume},
    scaledown_window=SCALEDOWN_WINDOW,
)
def run_pipeline_remote(request: dict) -> dict:
    data_volume.reload()

    phases = request.get("phases")
    if phases != list(PHASE_ORDER):
        raise ValueError(f"run_pipeline_remote currently supports only {list(PHASE_ORDER)}")

    output_subdir = request.get("output_subdir", "results/all")
    predictions_path = request.get("predictions_path", "")
    if not predictions_path:
        raise ValueError("pipeline requires predictions_path")

    pred_full = _volume_path(predictions_path)
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")

    phase1, call_acc_dir = _run_phase1(pred_full, output_subdir)
    phase2 = _run_phase2_on_call_acc(call_acc_dir, output_subdir)
    phase3 = _run_phase3_on_call_acc(call_acc_dir, output_subdir)

    data_volume.commit()
    return {
        "phases": list(PHASE_ORDER),
        "results": [phase1, phase2, phase3],
        "artifacts": _pipeline_artifacts(output_subdir),
    }


def _phase_request(
    phase: str,
    output_subdir: str,
    predictions: str = "",
    call_acc_subdir: str = "",
) -> dict:
    return {
        "phase": phase,
        "predictions_path": _upload_local_predictions(Path(predictions)) if phase == "phase1" else "",
        "call_acc_subdir": call_acc_subdir,
        "output_subdir": output_subdir,
    }


def _pipeline_request(predictions: str, output_subdir: str) -> dict:
    return {
        "phases": list(PHASE_ORDER),
        "predictions_path": _upload_local_predictions(Path(predictions)),
        "output_subdir": output_subdir,
    }


def _print_eval_json(summary: dict) -> None:
    print(EVAL_JSON_START)
    print(json.dumps(summary))
    print(EVAL_JSON_END)


@app.local_entrypoint()
def run_phase(
    phase: str,
    output_subdir: str,
    predictions: str = "",
    call_acc_subdir: str = "",
):
    summary = run_phase_remote.remote(_phase_request(phase, output_subdir, predictions, call_acc_subdir))
    _print_eval_json(summary)


@app.local_entrypoint()
def run_pipeline(
    predictions: str,
    output_subdir: str = "results/all",
):
    summary = run_pipeline_remote.remote(_pipeline_request(predictions, output_subdir))
    _print_eval_json(summary)
