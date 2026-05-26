# Deepbork Evaluation Refactor Plan

This plan defines the evaluation architecture as if it were being written from scratch. The goal is to replace phase-specific branching and legacy orchestration with a small, explicit pipeline model.

## Target Architecture

The evaluation system should have three layers:

1. `modal_eval_app.py`: TritonBench execution adapter.
2. `evaluate.py` or `evaluation/`: local pipeline orchestration, CLI, summary helpers.
3. `agentic_eval.py`: generation/repair attempt loop only.

The core rule is: phases are data, execution is generic, summaries are normalized.

## Public Behavior

Keep these CLI modes:

```bash
python3 evaluate.py --mode phase1 --predictions outputs/predictions.jsonl
python3 evaluate.py --mode phase2 --call-acc-subdir results/phase1/call_acc
python3 evaluate.py --mode phase3 --call-acc-subdir results/phase2/call_acc
python3 evaluate.py --mode all --predictions outputs/predictions.jsonl
```

Keep these agentic modes:

```bash
python3 agentic_eval.py --ops div --target-stage phase1
python3 agentic_eval.py --ops div --target-stage all
```

Do not support a Phase 1 -> Phase 2 only agentic target.

## Data Model

Create a normalized evaluation model before touching orchestration.

```python
@dataclass(frozen=True)
class PhaseSpec:
    name: str
    input_artifact: str
    output_artifact: str | None
    metric_key: str
    passed_files_key: str
    failed_files_key: str
    stdout_key: str
    stderr_key: str


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
```

Required phase registry:

```python
PHASES = {
    "phase1": PhaseSpec(...),
    "phase2": PhaseSpec(...),
    "phase3": PhaseSpec(...),
}
```

The registry is the only place that should know phase-specific summary field names.

## Modal Boundary

Refactor `modal_eval_app.py` first.

### Keep

TritonBench-specific implementation stays in `modal_eval_app.py`:

- Modal image.
- Modal volume.
- TritonBench repo setup and patches.
- `_run_call_acc`.
- `_run_exe_acc`.
- `_run_phase3_on_call_acc`.
- log writing.
- volume path validation and copying.

### Replace Public Remote API

Expose exactly two remote functions:

```python
@app.function(...)
def run_phase_remote(request: dict) -> dict:
    ...


@app.function(...)
def run_pipeline_remote(request: dict) -> dict:
    ...
```

Expose exactly two local entrypoints:

```python
@app.local_entrypoint()
def run_phase(...):
    ...


@app.local_entrypoint()
def run_pipeline(...):
    ...
```

Remove these legacy entrypoints and functions:

- `evaluate_phase1_only`
- `evaluate_phase2_only`
- `evaluate_phase3_only`
- `evaluate_through_phase2_entrypoint`
- `evaluate_all_entrypoint`
- any remote `evaluate_through_phase2`

### Modal Request Shape

Use one request shape:

```json
{
  "phase": "phase1",
  "phases": ["phase1", "phase2", "phase3"],
  "predictions_path": "uploads/predictions.jsonl",
  "call_acc_subdir": "results/phase1/call_acc",
  "output_subdir": "results/all"
}
```

For `run_phase_remote`, `phase` is required.

For `run_pipeline_remote`, `phases` is required and initially must be `["phase1", "phase2", "phase3"]`.

### Modal Result Shape

Every Modal phase should return a normalized phase result:

```json
{
  "phase": "phase1",
  "attempted_files": ["div.py"],
  "passed_files": ["div.py"],
  "failed_files": [],
  "metrics": {
    "passed": 1,
    "failed": 0,
    "rate": 100.0
  },
  "artifacts": {
    "artifacts_volume": "deepbork-phase1-data",
    "artifacts_subdir": "results/all",
    "call_acc_dir": "results/all/call_acc",
    "logs_dir": "results/all/logs"
  },
  "stdout_tail": "...",
  "stderr_tail": "",
  "failed": false
}
```

`run_pipeline_remote` should return:

```json
{
  "phases": ["phase1", "phase2", "phase3"],
  "results": [ ... normalized phase results ... ],
  "artifacts": {
    "artifacts_volume": "...",
    "artifacts_subdir": "...",
    "call_acc_dir": "...",
    "perf_results_dir": "...",
    "logs_dir": "..."
  }
}
```

The Modal layer should not emit legacy combined summaries. It should emit normalized results only.

## Pipeline Semantics

The full pipeline is:

```text
phase1(predictions.jsonl) -> call_acc/
phase2(call_acc/) -> pruned call_acc/
phase3(call_acc/) -> perf_results/
```

For `all`, run Phase 1, then Phase 2, then Phase 3 in one Modal function call.

Do not create a public Phase 1 -> Phase 2 only pipeline.

Standalone `phase2` and `phase3` are allowed only when the caller provides `--call-acc-subdir`.

## Evaluation Orchestration

After the Modal boundary is clean, refactor `evaluate.py`.

### Responsibilities

`evaluate.py` should own:

- CLI parsing.
- Mode validation.
- uploading local predictions through the Modal local entrypoint.
- calling either one phase or the full pipeline.
- returning the normalized evaluation summary used by the CLI and agentic repair loop.
- pass/fail helpers.

`evaluate.py` should not own:

- TritonBench subprocess details.
- Modal volume internals.
- Phase-specific execution code.

### Evaluation API

Expose these local Python functions:

```python
def evaluate_phase(
    phase: str,
    output_subdir: str,
    predictions: Path | None = None,
    call_acc_subdir: str | None = None,
) -> dict:
    ...


def evaluate_all(
    predictions: Path,
    output_subdir: str,
) -> dict:
    ...


def evaluate_local(
    mode: str,
    output_subdir: str,
    predictions: Path | None = None,
    call_acc_subdir: str | None = None,
) -> dict:
    ...
```

`evaluate_local(mode="all")` calls `evaluate_all`.

`evaluate_local(mode in {"phase1", "phase2", "phase3"})` calls `evaluate_phase`.

Any other mode is invalid.

### Canonical Summary Format

Use the normalized pipeline result as the canonical summary format:

```json
{
  "mode": "all",
  "failed_phase": "phase2",
  "passed_through": "phase1",
  "artifacts": {
    "artifacts_volume": "deepbork-phase1-data",
    "artifacts_subdir": "results/all",
    "call_acc_dir": "results/all/call_acc",
    "perf_results_dir": "results/all/perf_results",
    "logs_dir": "results/all/logs"
  },
  "results": [
    {
      "phase": "phase1",
      "attempted_files": ["div.py"],
      "passed_files": ["div.py"],
      "failed_files": [],
      "metrics": {
        "passed": 1,
        "failed": 0,
        "rate": 100.0
      },
      "stdout_tail": "...",
      "stderr_tail": "",
      "failed": false
    },
    {
      "phase": "phase2",
      "attempted_files": ["div.py"],
      "passed_files": [],
      "failed_files": ["div.py"],
      "metrics": {
        "passed": 0,
        "failed": 1,
        "rate": 0.0
      },
      "stdout_tail": "...",
      "stderr_tail": "...",
      "failed": true
    }
  ]
}
```

Do not emit or preserve legacy flattened phase/result fields as the main contract:

- `phase1_call_acc`
- `phase2_exec_acc`
- `phase3_efficiency`
- `phase1_passed_files`
- `phase2_passed_files`
- `phase3_passed_files`
- top-level `stdout_tail`
- top-level `stderr_tail`
- top-level `call_acc_dir`
- top-level `perf_results_dir`

If an external script still requires those fields, add a separate short-lived compatibility command or helper, but do not make it part of the core evaluation path.

## Agentic Loop

Refactor `agentic_eval.py` last.

### Responsibilities

`agentic_eval.py` should only:

- select one operator.
- generate the initial candidate.
- write attempt artifacts.
- call `evaluate_local(mode="phase1"|"all")`.
- decide pass/fail using helpers from `evaluate.py`.
- build repair context based on the normalized `failed_phase`.
- build repair context from the failed phase result in `summary["results"]`.
- generate repaired candidates.

### Remove

Remove all evaluation mapping logic from `agentic_eval.py`:

- no Phase 2 target.
- no Phase 3 target.
- no target-stage-to-phase-list mapping.
- no direct knowledge of Modal entrypoints.

### Agentic Target Mapping

Use only:

```python
AGENTIC_TARGETS = ("phase1", "all")
PASS_STAGE = {
    "phase1": "phase1",
    "all": "phase3",
}
```

`--target-stage all` means the candidate must pass Phase 3.

Failures inside the full pipeline can still be `phase1`, `phase2`, or `phase3` for repair-rule selection.

## Implementation Order

### Step 1: Add Normalized Models

Add the dataclasses and phase registry in `evaluate.py` or a new module:

```text
evaluation/model.py
```

No behavior change yet.

### Step 2: Normalize Modal Phase Results

In `modal_eval_app.py`, create helper functions:

```python
def normalize_phase1_summary(summary: dict) -> dict: ...
def normalize_phase2_summary(summary: dict) -> dict: ...
def normalize_phase3_summary(summary: dict) -> dict: ...
```

Use these to return normalized phase results from Modal.

### Step 3: Replace Modal Public API

Add:

- `run_phase_remote`
- `run_pipeline_remote`
- `run_phase`
- `run_pipeline`

Keep old entrypoints temporarily only until `evaluate.py` is migrated.

### Step 4: Migrate `evaluate.py`

Change `evaluate.py` so it calls only:

- `modal_eval_app.py::run_phase`
- `modal_eval_app.py::run_pipeline`

At this point, `evaluate.py` must not call old Modal entrypoints.

### Step 5: Delete Legacy Modal Entrypoints

Delete:

- phase-specific local entrypoints.
- Phase 1 -> Phase 2 local entrypoint.
- any multi-phase remote function other than `run_pipeline_remote`.

### Step 6: Replace `evaluate.py` Summary Combining

Delete scattered phase-specific summary merging and return the normalized summary format directly:

```python
def evaluation_summary(mode: str, results: list[dict]) -> dict:
    ...
```

This function computes only cross-phase fields:

- `mode`
- `failed_phase`
- `passed_through`
- top-level `artifacts`
- ordered `results`

It should not produce legacy flattened phase keys.

### Step 7: Migrate `agentic_eval.py`

Change `agentic_eval.py` to call:

```python
summary = evaluate_local(
    mode=args.target_stage,
    predictions=prediction_path,
    output_subdir=output_subdir,
)
```

Restrict `--target-stage` to:

```text
phase1
all
```

### Step 8: Delete Dead Compatibility

Search and remove all references to:

```text
phase1_phase2
evaluate_phase1_only
evaluate_phase2_only
evaluate_phase3_only
evaluate_through_phase2
evaluate_through
evaluate_target_stage
```

## Verification

Static checks:

```bash
python3 -m py_compile evaluate.py agentic_eval.py modal_eval_app.py
python3 evaluate.py --help
python3 agentic_eval.py --help
rg "phase1_phase2|evaluate_phase1_only|evaluate_phase2_only|evaluate_phase3_only|evaluate_through_phase2|evaluate_through|evaluate_target_stage" .
```

Modal smoke checks:

```bash
python3 evaluate.py --mode phase1 --predictions outputs/predictions.jsonl
python3 evaluate.py --mode all --predictions outputs/predictions.jsonl
```

Standalone phase checks after Phase 1 has produced artifacts:

```bash
python3 evaluate.py --mode phase2 --call-acc-subdir results/phase1/call_acc
python3 evaluate.py --mode phase3 --call-acc-subdir results/phase2/call_acc
```

Agentic checks:

```bash
python3 agentic_eval.py --ops div --target-stage phase1 --max-attempts 1
python3 agentic_eval.py --ops div --target-stage all --max-attempts 1
```

## Acceptance Criteria

- `modal_eval_app.py` has one public single-phase API and one public full-pipeline API.
- No public Phase 1 -> Phase 2 only pipeline exists.
- `evaluate.py` supports only `phase1`, `phase2`, `phase3`, and `all`.
- `agentic_eval.py` supports only `phase1` and `all`.
- Phase-specific field names are isolated to the phase registry and normalized phase results.
- The agentic repair loop reads:
  - `failed_phase`
  - `passed_through`
  - `artifacts.call_acc_dir`
  - `artifacts.perf_results_dir`
  - failed phase `stdout_tail`
  - failed phase `stderr_tail`
  - per-phase `passed_files`
  - per-phase `failed_files`
