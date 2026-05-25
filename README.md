# deepbork

Agentic TritonBench-T code generation and evaluation orchestrator for
Deepbork.

The repo is no longer aimed only at producing a `predictions.jsonl` file.
`predictions.jsonl` is still the batch interchange format, but the main target
is now the full local-to-Modal evaluation loop:

1. generate Triton Python candidates from TritonBench-T prompts
2. evaluate candidates on Modal using TritonBench-T Phase 1 and Phase 2
3. persist local and remote artifacts per run or per agentic attempt
4. repair failed candidates with evaluation feedback and curated rules
5. keep each evaluation phase callable independently so future Phase 3 and
   phase-specific agentic policies can be added without changing the whole
   pipeline

The current implemented pipeline supports generation, Phase 1 call accuracy,
Phase 2 execution accuracy, and a one-operator agentic repair loop. Phase 3
performance benchmarking is the next planned evaluation stage.

## Repository Flow

The main files have separate responsibilities:

```text
main.py             batch code generation; writes outputs/predictions.jsonl
evaluate.py         local CLI/Python launcher for Modal evaluation
modal_eval_app.py   Modal image, volume, GPU execution, Phase 1/2 evaluators
agentic_eval.py     generate/evaluate/repair loop for one operator
tritonbench_helpers.py
                    metadata matching between prompts, ops, and predictions
repair_rules/       editable repair knowledge injected into repair prompts
```

Batch evaluation flow:

```text
main.py
  -> outputs/predictions.jsonl
  -> evaluate.py --mode phase1|phase2|all
  -> modal_eval_app.py
  -> Modal volume results/
```

Agentic flow:

```text
agentic_eval.py
  -> generate one candidate using main.py prompt helpers
  -> write outputs/agentic_eval/<op>/attempt_N/predictions.jsonl
  -> evaluate through target stage on Modal
  -> accept, or repair using stdout/stderr + repair_rules/
```

The agentic target is a stage threshold:

```text
--target-stage phase1  -> run Phase 1 only and accept Phase 1 survivors
--target-stage phase2  -> run Phase 1 then Phase 2 and accept Phase 2 survivors
```

The standalone evaluator mode is a direct Modal command:

```text
--mode phase1  -> predictions.jsonl -> call_acc/
--mode phase2  -> existing call_acc/ -> Phase 2 surviving call_acc/
--mode all     -> predictions.jsonl -> Phase 1 -> Phase 2
```

## Contract

The LLM endpoint is called through the OpenAI Python client with a custom
`base_url`:

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="https://your-workspace--example-vllm-inference-serve.modal.run/v1",
)

completion = client.chat.completions.create(
    model="llm",
    messages=[
        {"role": "system", "content": "PROMPT_HEADER"},
        {"role": "user", "content": "instruction_or_instruction_plus_input"},
    ],
)
```

On the wire, this is still an OpenAI-compatible chat completion request:

```json
{
  "model": "llm",
  "stream": false,
  "messages": [
    {"role": "system", "content": "PROMPT_HEADER"},
    {"role": "user", "content": "instruction_or_instruction_plus_input"}
  ]
}
```

The endpoint returns raw generated text. `deepbork` applies the same code-fence
cleanup logic used by the benchmark harness and writes JSONL records:

```json
{"instruction": "...", "predict": "..."}
```

The generated JSONL file is the input contract for batch evaluation. Phase 1
consumes it directly; Phase 2 can either run immediately after Phase 1
(`--mode all`) or later from a Modal-volume `call_acc/` directory.

## Dataset Files

This repo expects the Alpaca prompt files to exist locally:

```text
data/
  TritonBench_T_simp_alpac_v1.json
  TritonBench_T_comp_alpac_v1.json
  TritonBench_T_v1.jsonl
```

No full TritonBench clone is needed for generation.

Evaluation uses `modal_eval_app.py`, which clones the upstream TritonBench repo
inside the Modal image and patches the Phase 1 and Phase 2 script paths needed
for unattended execution. The app also includes
`tritonbench_helpers.py` in the remote container with Modal's local Python
source packaging so Phase 1 reporting reuses the same metadata matching logic
as `--ops`.

## TritonBench-T File Roles

The source of truth for these benchmark assets is the upstream
[thunlp/TritonBench](https://github.com/thunlp/TritonBench) repo. This project
uses the TritonBench-T track: PyTorch-to-Triton translation. Inputs describe
PyTorch-style operators and ask the model to generate Triton wrapper functions.

Relevant files in the benchmark flow:

```text
Input prompts:
  TritonBench_T_simp_alpac_v1.json
  TritonBench_T_comp_alpac_v1.json

Metadata/index:
  TritonBench_T_v1.jsonl

Golden test/reference files:
  TritonBench_T_v1/tanh.py
  TritonBench_T_v1/fused_bmm_rmsnorm_gelu_dropout_sub.py
  ...

Generated output:
  predictions.jsonl
```

`deepbork` uses the Alpaca prompt files for code generation. The TritonBench
evaluator uses the `instruction` field in `predictions.jsonl` to recover the
matching metadata row from `TritonBench_T_v1.jsonl`, then appends tests from the
corresponding golden file in `TritonBench_T_v1/` to the generated `predict`
code.

## Setup

Install the lightweight orchestrator dependency:

```bash
python3 -m pip install -r requirements.txt
```

Create a `.env` file or export the endpoint in your shell:

```bash
LLM_PROVIDER=modal-vllm
DEFAULT_ENDPOINT=https://your-workspace--example-vllm-inference-serve.modal.run
VLLM_MODEL=llm
VLLM_API_KEY=EMPTY
```

The endpoint should be the base URL from `modal_vllm`, without
`/v1/chat/completions`. `deepbork` accepts either the base endpoint or the
endpoint ending in `/v1`.

## Main Commands

For the current repo direction, the most important commands are:

```bash
# Agentic one-operator loop, accepting only Phase 1 survivors.
python3 agentic_eval.py --provider modal-vllm --ops div

# Agentic one-operator loop, accepting only candidates that pass Phase 1 and Phase 2.
python3 agentic_eval.py --provider modal-vllm --ops div --target-stage phase2

# Batch generation, still useful for offline or bulk evaluation.
python3 main.py --provider modal-vllm --ops div

# Batch evaluation through Phase 1 and Phase 2.
python3 evaluate.py --mode all --predictions outputs/predictions.jsonl

# Standalone Phase 2 from the default Phase 1 call_acc directory.
python3 evaluate.py --mode phase2
```

## Batch Generation

Smoke test one item with `modal_vllm`:

```bash
python3 main.py --provider modal-vllm --limit 1
```

Generate a specific operator by TritonBench-T filename stem:

```bash
python3 main.py --provider openai --ops tanh
```

Multiple operators can be passed as a comma-separated list, without the `.py`
extension:

```bash
python3 main.py --provider openai --ops tanh,sqrt,fused_bmm_rmsnorm_gelu_dropout_sub
```

`main.py` reads from `data/`, uses the `simp` dataset by default, resolves the
endpoint/API key from environment variables, and writes
`outputs/predictions.jsonl`. These are repo defaults, not CLI options.

`--ops` uses `data/TritonBench_T_v1.jsonl` as the metadata index. It matches the
requested filename stem to the metadata `file` field, then matches that row's
description to the Alpaca prompt's `Functional Description`. If no `--limit` or
`--ops` is provided, `deepbork` generates the first 3 prompt rows by default. If
both `--limit` and `--ops` are provided, `--limit` wins and `--ops` is ignored.
Use `--limit 0` to generate all prompt rows.

To change the default dataset or output path for repo development, edit
`DEFAULT_DATASET` or `DEFAULT_OUTPUT_PATH` in `main.py`.

Use the real OpenAI API instead:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

python3 main.py \
  --provider openai \
  --limit 1 \
  --max-tokens 512
```

The generated file is written to `outputs/predictions.jsonl` by default.

## Batch Evaluation on Modal

Install dependencies and authenticate Modal once:

```bash
python3 -m pip install -r requirements.txt
modal setup
```

Generate predictions locally, then run Phase 1 call accuracy on Modal:

```bash
python3 main.py --provider modal-vllm --limit 1

python3 evaluate.py --mode phase1 --predictions outputs/predictions.jsonl
```

`evaluate.py` is a thin batch wrapper around:

```bash
modal run modal_eval_app.py::evaluate_phase1_only \
  --predictions outputs/predictions.jsonl
```

Run Phase 1 followed by Phase 2 execution accuracy from a local
`predictions.jsonl`:

```bash
python3 evaluate.py --mode all --predictions outputs/predictions.jsonl
```

Run Phase 2 later from an existing Phase 1 `call_acc` folder in the Modal
Volume:

```bash
python3 evaluate.py --mode phase2
```

By default this reads `results/phase1/call_acc` from the Modal Volume and writes
Phase 2 artifacts to `results/phase2`. Pass `--call-acc-subdir` only for a
different `call_acc/` directory, such as an agentic attempt:

```bash
python3 evaluate.py \
  --mode phase2 \
  --call-acc-subdir results/eval/agentic/div/attempt_003/call_acc
```

This uses the same Modal image. Phase 2 consumes the `.py` files that survive
Phase 1 in `call_acc/`, copies them into the Phase 2 output directory when run
standalone, runs `1_exe_acc.py::execute_4folder`, and deletes files whose
outputs differ from the golden implementation.

The Modal app uploads the local JSONL into the `deepbork-phase1-data` volume,
runs the requested phases on a single GPU, and prints a JSON summary:

```json
{
  "total_predictions": 1,
  "phase1_call_acc": {"passed": 0, "failed": 1, "rate": 0.0},
  "attempted_files": ["tanh.py"],
  "failed_phase": "phase1",
  "phase1_passed_files": [],
  "phase1_failed_files": ["tanh.py"],
  "artifacts_volume": "deepbork-phase1-data",
  "artifacts_subdir": "results/phase1",
  "call_acc_dir": "results/phase1/call_acc"
}
```

Operators that pass Phase 1 are written as `.py` files under
`results/phase1/call_acc/` in the Modal Volume. Phase 1 stdout and stderr are
also written under `results/phase1/logs/`. Download them with:

```bash
modal volume get deepbork-phase1-data results/phase1 ./local-phase1-results/
```

Runtime knobs:

```bash
DEEPBORK_EVAL_GPU=A10 modal run modal_eval_app.py::evaluate_phase1_only \
  --predictions outputs/predictions.jsonl

DEEPBORK_MODAL_VOLUME=my-volume modal run modal_eval_app.py::evaluate_phase1_only \
  --predictions outputs/predictions.jsonl
```

This evaluation surface is the deterministic hook used by the agentic loop:
write one candidate prediction, call Modal evaluation, inspect `failed_phase`
and the phase-specific passed/failed files, then repair or accept.

## Agentic Evaluation

Run a one-operator repair loop targeting Phase 1:

```bash
python3 agentic_eval.py \
  --provider modal-vllm \
  --ops div \
  --max-attempts 5
```

Require Phase 1 and Phase 2 to pass before accepting:

```bash
python3 agentic_eval.py \
  --provider modal-vllm \
  --ops div \
  --target-stage phase2 \
  --max-attempts 5
```

The loop:

- selects one benchmark item via the same `--ops` metadata matching used by
  `main.py`
- generates an initial prediction with the normal prompt
- writes one local `predictions.jsonl` per attempt
- runs Modal evaluation through the requested target stage
- if evaluation fails, asks the model to repair the previous code using
  `stdout_tail` and `stderr_tail`
- injects matching curated repair rules from the Phase 1 or Phase 2 rules file,
  depending on `failed_phase`

Local attempt artifacts are written under:

```text
outputs/agentic_eval/<op>/
├── attempt_001/
│   ├── predictions.jsonl
│   ├── predict.py
│   ├── evaluation_summary.json
│   └── repair_prompt.json  # present when a repair attempt is needed
└── result.json
```

Remote Modal artifacts use unique per-attempt directories:

```text
results/eval/agentic/<op>/attempt_001/
```

Curated repair rules are data, not code. Add new recurring Phase 1 failure
patterns to:

```text
repair_rules/triton_phase1_rules.json
```

Add Phase 2 semantic mismatch rules to:

```text
repair_rules/triton_phase2_rules.json
```

Each rule has match patterns over `previous_predict`, `phase1_output`,
`phase2_output`, or `evaluation_output`, plus a problem description, fix, and
avoid list. The agentic loop loads those default rule files and includes only
the matching rules for the failed phase in the next repair prompt.

## Current Implementation

Implemented now:

- read `data/TritonBench_T_<simp|comp>_alpac_v1.json`
- build prompt messages
- call either `modal_vllm` or OpenAI through the `openai` client
- write `predictions.jsonl`
- upload a local `predictions.jsonl` to Modal
- run TritonBench-T Phase 1 and Phase 2
- keep Phase 1 and Phase 2 callable independently
- run a simple agentic repair loop through a target stage
- return attempted, passed, failed, and failed-stage information

Planned next stages:

- add Phase 3 performance benchmarking as another modular evaluation stage
- integrate XGrammar or equivalent local generation constraints
- make phase-specific agentic policies easier to compose, e.g. agentic Phase 1
  followed by deterministic Phase 2
- expand repair rules from repeated Phase 1 and Phase 2 failures
