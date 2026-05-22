"""Modal evaluator for deepbork predictions.

This app keeps TritonBench-T evaluation phases independently callable:

- Phase 1: upload/use a predictions.jsonl and run call accuracy.
- Phase 2: use an existing Modal-volume call_acc folder and run execution accuracy.
- All: run Phase 1, then Phase 2 on the Phase 1 survivors.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import modal

from tritonbench_helpers import DEFAULT_METADATA_FILE, load_metadata, prediction_files


APP_NAME = "deepbork-eval"
TRITONBENCH_REPO = "https://github.com/thunlp/TritonBench.git"

DEFAULT_GPU = os.environ.get("DEEPBORK_EVAL_GPU", os.environ.get("DEEPBORK_PHASE1_GPU", "T4"))
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

PATCH_EXE_ACC = (
    f"""sed -i """
    f"""-e 's|^gold_folder = .*|gold_folder = "{REPO_DIR}/data/TritonBench_T_v1/"|' """
    f"""-e 's|^py_interpreter = .*|import sys; py_interpreter = sys.executable|' """
    f"""{REPO_DIR}/EVAL/eval_T/1_exe_acc.py"""
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
    .run_commands(PATCH_CALL_ACC, PATCH_EXE_ACC)
    .run_commands(
        f"ln -s {REPO_DIR}/EVAL/eval_T/0_call_acc.py {REPO_DIR}/EVAL/eval_T/call_acc.py",
        f"ln -s {REPO_DIR}/EVAL/eval_T/1_exe_acc.py {REPO_DIR}/EVAL/eval_T/exe_acc.py",
    )
    .add_local_python_source("tritonbench_helpers")
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


def _phase1_summary(
    output_subdir: str,
    attempted: list[str],
    passed: list[str],
    failed: list[str],
    stdout_text: str,
    stderr_text: str,
) -> dict:
    total = len(attempted)
    return {
        "total_predictions": total,
        "phase1_call_acc": {
            "passed": len(passed),
            "failed": len(failed),
            "rate": round(100 * len(passed) / total, 2) if total else 0,
        },
        "failed_phase": "phase1" if failed else None,
        "attempted_files": attempted,
        "passed_files": passed,
        "failed_files": failed,
        "phase1_passed_files": passed,
        "phase1_failed_files": failed,
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
        "call_acc_dir": f"{output_subdir}/call_acc",
        "logs_dir": f"{output_subdir}/logs",
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
        "phase1_stdout_tail": stdout_text[-4000:],
        "phase1_stderr_tail": stderr_text[-4000:],
    }


def _phase2_summary(
    output_subdir: str,
    candidate_files: list[str],
    passed: list[str],
    failed: list[str],
    stdout_text: str,
    stderr_text: str,
) -> dict:
    total = len(candidate_files)
    return {
        "total_predictions": total,
        "phase2_exec_acc": {
            "passed": len(passed),
            "failed": len(failed),
            "rate": round(100 * len(passed) / total, 2) if total else 0,
        },
        "failed_phase": "phase2" if failed else None,
        "attempted_files": candidate_files,
        "passed_files": passed,
        "failed_files": failed,
        "phase2_passed_files": passed,
        "phase2_failed_files": failed,
        "artifacts_volume": VOLUME_NAME,
        "artifacts_subdir": output_subdir,
        "call_acc_dir": f"{output_subdir}/call_acc",
        "logs_dir": f"{output_subdir}/logs",
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
        "phase2_stdout_tail": stdout_text[-4000:],
        "phase2_stderr_tail": stderr_text[-4000:],
    }


def _run_phase1(predictions_path: Path, output_subdir: str) -> tuple[dict, Path]:
    _, call_acc_dir, logs_dir = _prepare_output(output_subdir, reset_call_acc=True)
    attempted = _prediction_files(predictions_path)

    print("\n" + "=" * 70 + "\n=== Phase 1: call accuracy ===\n" + "=" * 70, flush=True)
    stdout_text, stderr_text = _run_call_acc(predictions_path, call_acc_dir)
    _write_phase_logs(logs_dir, "phase1", stdout_text, stderr_text)

    passed = sorted(path.name for path in call_acc_dir.glob("*.py"))
    passed_set = set(passed)
    failed = [file_name for file_name in attempted if file_name not in passed_set]
    return _phase1_summary(output_subdir, attempted, passed, failed, stdout_text, stderr_text), call_acc_dir


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
    return _phase2_summary(output_subdir, candidate_files, passed, failed, stdout_text, stderr_text)


def _copy_call_acc_for_phase2(call_acc_subdir: str, output_subdir: str) -> Path:
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


def _combine_phase_summaries(phase1: dict, phase2: dict | None) -> dict:
    if phase2 is None:
        return phase1

    stdout_tail = (phase1.get("phase1_stdout_tail", "") + phase2.get("phase2_stdout_tail", ""))[-4000:]
    stderr_tail = (phase1.get("phase1_stderr_tail", "") + phase2.get("phase2_stderr_tail", ""))[-4000:]
    failed_phase = phase1.get("failed_phase") or phase2.get("failed_phase")

    combined = dict(phase1)
    combined.update(
        {
            "failed_phase": failed_phase,
            "phase2_exec_acc": phase2["phase2_exec_acc"],
            "phase2_passed_files": phase2["phase2_passed_files"],
            "phase2_failed_files": phase2["phase2_failed_files"],
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
            "phase2_stdout_tail": phase2["phase2_stdout_tail"],
            "phase2_stderr_tail": phase2["phase2_stderr_tail"],
        }
    )
    return combined


def _upload_local_predictions(local_path: Path) -> str:
    if not local_path.exists():
        raise FileNotFoundError(local_path)

    remote = f"uploads/{local_path.name}"
    print(f"uploading {local_path} -> volume://{remote}", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_path), remote)
    return remote


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60 * 3, volumes={DATA_DIR: data_volume})
def evaluate_phase1(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results/phase1",
) -> dict:
    pred_full = _volume_path(predictions_path)
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")
    summary, _ = _run_phase1(pred_full, output_subdir)
    data_volume.commit()
    return summary


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60 * 3, volumes={DATA_DIR: data_volume})
def evaluate_phase2(
    call_acc_subdir: str,
    output_subdir: str = "results/phase2",
) -> dict:
    call_acc_dir = _copy_call_acc_for_phase2(call_acc_subdir, output_subdir)
    summary = _run_phase2_on_call_acc(call_acc_dir, output_subdir)
    data_volume.commit()
    return summary


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60 * 3, volumes={DATA_DIR: data_volume})
def evaluate_all(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results/phase1_phase2",
) -> dict:
    pred_full = _volume_path(predictions_path)
    if not pred_full.exists():
        raise FileNotFoundError(f"predictions file not found in volume: {pred_full}")

    phase1, call_acc_dir = _run_phase1(pred_full, output_subdir)
    phase2 = None
    if phase1["phase1_passed_files"]:
        phase2 = _run_phase2_on_call_acc(call_acc_dir, output_subdir)
    else:
        logs_dir = _volume_path(output_subdir) / "logs"
        _write_phase_logs(logs_dir, "phase2", "skipped Phase 2: no Phase 1 survivors\n", "")
        phase2 = _phase2_summary(output_subdir, [], [], [], "skipped Phase 2: no Phase 1 survivors\n", "")

    summary = _combine_phase_summaries(phase1, phase2)
    data_volume.commit()
    return summary


@app.local_entrypoint()
def evaluate_phase1_only(
    predictions: str,
    output_subdir: str = "results/phase1",
):
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate_phase1.remote(predictions_path=remote, output_subdir=output_subdir)
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate_phase2_only(
    call_acc_subdir: str,
    output_subdir: str = "results/phase2",
):
    summary = evaluate_phase2.remote(
        call_acc_subdir=call_acc_subdir,
        output_subdir=output_subdir,
    )
    print(json.dumps(summary, indent=2))


@app.local_entrypoint()
def evaluate_all_entrypoint(
    predictions: str,
    output_subdir: str = "results/phase1_phase2",
):
    remote = _upload_local_predictions(Path(predictions))
    summary = evaluate_all.remote(predictions_path=remote, output_subdir=output_subdir)
    print(json.dumps(summary, indent=2))
