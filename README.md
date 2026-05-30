# deepbork

Deepbork is an agentic-first TritonBench-T executor.

Its primary workflow is to generate a Triton candidate for one operator, evaluate
that candidate on Modal with the TritonBench-T harness, repair the candidate
from phase-specific feedback when it fails, and repeat until the target passes
or the attempt budget is exhausted.

Batch agentic runs keep each operator isolated. Every selected operator runs
the same workflow separately, and every attempt writes its own one-row
`predictions.jsonl`, generated module, and evaluation summary.

## What Deepbork Runs

Deepbork targets the TritonBench-T PyTorch-to-Triton translation track. Each
operator prompt asks the model to produce a complete Python module containing
the required wrapper function and any Triton kernels it needs.

Evaluation has three TritonBench-T phases:

```text
phase1  predictions.jsonl -> call_acc/      runtime and call accuracy
phase2  call_acc/          -> pruned call_acc/ execution accuracy
phase3  call_acc/          -> perf_results/ efficiency benchmark
```

Agentic acceptance targets are intentionally small:

```text
--target-stage phase1  accept candidates that pass Phase 1
--target-stage all     accept candidates that pass Phase 1, Phase 2, and Phase 3
```

## Agentic Workflow

The main flow is `agentic_eval.py`:

```text
agentic_eval.py
  -> select one or more TritonBench-T operators
  -> generate one candidate for the current operator
  -> write outputs/agentic_eval/<op>/attempt_N/predictions.jsonl
  -> evaluate Phase 1 or the full Phase 1+2+3 pipeline on Modal
  -> inspect the normalized evaluation summary
  -> accept the candidate, or repair using the failed phase result
  -> repeat up to --max-attempts
```

Every repair attempt starts again from Phase 1. A source change creates a fresh
Phase 1, Phase 2, and Phase 3 evaluation path for that candidate.

## Quickstart

### Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Create a local environment file from the documented template:

```bash
cp .env.example .env
```

Edit `.env` with your provider, endpoint, and API keys. The template includes
placeholders for Modal vLLM and OpenAI:

```dotenv
LLM_PROVIDER=modal-vllm
DEFAULT_ENDPOINT=https://your-workspace--your-vllm-app.modal.run
VLLM_MODEL=llm
VLLM_API_KEY=EMPTY

OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
```

Authenticate Modal and deploy the evaluation app:

```bash
modal setup
modal deploy modal_eval_app.py
```

### Basic Commands

Run one operator through Phase 1:

```bash
python3 agentic_eval.py --ops div --target-stage phase1
```

Run one operator through the full pipeline:

```bash
python3 agentic_eval.py --ops div --target-stage all
```

## How To Run Agentic Evaluation

Run explicit operators:

```bash
python3 agentic_eval.py \
  --ops div,tanh,sqrt \
  --target-stage all \
  --max-attempts 5
```

Run the first N dataset items:

```bash
python3 agentic_eval.py \
  --limit 10 \
  --target-stage phase1 \
  --max-attempts 3
```

Run all dataset items:

```bash
python3 agentic_eval.py \
  --limit 0 \
  --target-stage all
```

Selection rules:

- `--ops` accepts comma-separated operator filename stems, without `.py`.
- `--limit N` selects the first N prompt rows from the default dataset.
- `--limit 0` selects all prompt rows.
- if both `--ops` and `--limit` are provided, `--limit` wins.
- if neither is provided, `agentic_eval.py` runs one dataset item.

Use OpenAI by setting `LLM_PROVIDER=openai` in `.env`, or override the provider
for one command:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

python3 agentic_eval.py \
  --provider openai \
  --ops div \
  --target-stage phase1
```

## What Agentic Evaluation Generates

Local artifacts are written per operator and per attempt:

```text
outputs/agentic_eval/
├── batch_result.json
└── <op>/
    ├── result.json
    └── attempt_001/
        ├── predictions.jsonl
        ├── predict.py
        ├── evaluation_summary.json
        └── repair_prompt.json  # present only when a repair is requested
```

`predictions.jsonl` inside an attempt has exactly one JSONL record:

```json
{"instruction": "...", "predict": "..."}
```

Remote Modal artifacts are also isolated per operator and attempt:

```text
results/eval/agentic/<op>/attempt_001/
  call_acc/
  perf_results/
  logs/
```

Download artifacts from the Modal volume with:

```bash
modal volume get deepbork-data \
  results/eval/agentic/div/attempt_001 \
  ./local-agentic-div-attempt-001/
```

## Evaluation Summary Format

Evaluation returns a normalized summary:

```json
{
  "mode": "all",
  "failed_phase": "phase2",
  "passed_through": "phase1",
  "artifacts": {
    "artifacts_volume": "deepbork-data",
    "artifacts_subdir": "results/eval/agentic/div/attempt_001",
    "call_acc_dir": "results/eval/agentic/div/attempt_001/call_acc",
    "perf_results_dir": "results/eval/agentic/div/attempt_001/perf_results",
    "logs_dir": "results/eval/agentic/div/attempt_001/logs"
  },
  "results": [
    {
      "phase": "phase1",
      "attempted_files": ["div.py"],
      "passed_files": ["div.py"],
      "failed_files": [],
      "metrics": {"passed": 1, "failed": 0, "rate": 100.0},
      "stdout_tail": "...",
      "stderr_tail": "",
      "failed": false
    },
    {
      "phase": "phase2",
      "attempted_files": ["div.py"],
      "passed_files": [],
      "failed_files": ["div.py"],
      "metrics": {"passed": 0, "failed": 1, "rate": 0.0},
      "stdout_tail": "...",
      "stderr_tail": "...",
      "failed": true
    }
  ]
}
```

Repair prompts use the failed phase result from `results[]`, plus the top-level
artifact paths and `passed_through` value.

## Modal Execution Model

`modal_eval_app.py` owns TritonBench-specific remote execution:

- image setup
- TritonBench clone and script patches
- Modal volume access
- GPU execution
- Phase 1 call accuracy
- Phase 2 execution accuracy
- Phase 3 efficiency benchmarking
- phase logs and artifact directories

It exposes two public execution surfaces:

```text
run_phase     one standalone phase through `modal run`
run_pipeline  full Phase 1+2+3 pipeline through `modal run`
```

`evaluate.py --mode all` calls the deployed `run_pipeline_remote` function.
Standalone `phase1`, `phase2`, and `phase3` use `modal run` against
`modal_eval_app.py::run_phase`.

## Dataset And Operator Selection

Deepbork reads these local files:

```text
data/
  TritonBench_T_simp_alpac_v1.json
  TritonBench_T_comp_alpac_v1.json
  TritonBench_T_v1.jsonl
```

The configured dataset is `simp`.

Operator selection uses `data/TritonBench_T_v1.jsonl` as the metadata index.
`--ops div,tanh` matches each requested filename stem to the metadata `file`
field, then matches that metadata row's description to an Alpaca prompt's
`Functional Description` block.

`--limit` selects prompt rows directly from the dataset, then maps each row back
to its TritonBench-T filename through metadata.

Generation uses the local prompt and metadata files. Evaluation clones and
patches the upstream TritonBench repo inside the Modal image.

## Lower-Level Tools

The agentic loop is the main interface. These lower-level tools are useful for
debugging, smoke tests, and non-agentic experiments.

Generate a raw batch `outputs/predictions.jsonl`:

```bash
python3 main.py --limit 1
python3 main.py --ops tanh,sqrt
```

`main.py` writes a single JSONL file containing all selected predictions.
Agentic batch mode writes one attempt-local JSONL file per operator attempt.

Run manual evaluation:

```bash
# Phase 1 from a local predictions file.
python3 evaluate.py --mode phase1 --predictions outputs/predictions.jsonl

# Full Phase 1+2+3 pipeline from a local predictions file.
python3 evaluate.py --mode all --predictions outputs/predictions.jsonl

# Phase 2 from an existing Modal-volume call_acc directory.
python3 evaluate.py --mode phase2 --call-acc-subdir results/phase1/call_acc

# Phase 3 from an existing Modal-volume call_acc directory.
python3 evaluate.py --mode phase3 --call-acc-subdir results/phase2/call_acc
```


## Repository Layout

```text
agentic_eval.py       primary generate/evaluate/repair executor
evaluate.py           local evaluation CLI/API and normalized summary helpers
modal_eval_app.py     Modal TritonBench-T execution backend
main.py               raw batch prediction generation utility
tritonbench_helpers.py
                      metadata, operator, and prediction matching helpers
evaluation/model.py   normalized phase specs and result/context dataclasses
constants.py          shared app names, paths, sentinels, and output subdirs
data/                 local TritonBench-T prompts and metadata
outputs/              local generated artifacts
```

## Troubleshooting

- `--target-stage all` requires the Modal app to be deployed because it calls
  `run_pipeline_remote`.
- `phase2` and `phase3` standalone evaluation require an existing
  Modal-volume `call_acc/` directory.
- agentic batch mode is sequential; it runs a complete isolated agentic flow
  for each selected operator.
- generated `outputs/` artifacts are runtime output and should not be committed
  unless intentionally curated.
