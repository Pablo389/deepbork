# deepbork

Local prediction-generation orchestrator for Deepbork.

This repo only creates a `predictions.jsonl` file. It does not run local model
inference, build a Modal image, execute TritonBench evaluation, or upload
artifacts anywhere.

The current flow is:

1. read the local TritonBench-T Alpaca dataset files from `data/`
2. build OpenAI-style chat messages for each benchmark item
3. call either a deployed `modal_vllm` endpoint or the real OpenAI API with the official `openai` client
4. clean the returned code block
5. write one JSON line per generated operator

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

The resulting file is the only artifact this repo is responsible for.

## Dataset Files

This repo expects the Alpaca prompt files to exist locally:

```text
data/
  TritonBench_T_simp_alpac_v1.json
  TritonBench_T_comp_alpac_v1.json
  TritonBench_T_v1.jsonl
```

No full TritonBench clone is needed for generation.

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

## Generate Predictions

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

`--ops` uses `data/TritonBench_T_v1.jsonl` as the metadata index. It matches the
requested filename stem to the metadata `file` field, then matches that row's
description to the Alpaca prompt's `Functional Description`. If no `--limit` or
`--ops` is provided, `deepbork` generates the first 3 prompt rows by default. If
both `--limit` and `--ops` are provided, `--limit` wins and `--ops` is ignored.
Use `--limit 0` to generate all prompt rows.

Pass the `modal_vllm` endpoint explicitly:

```bash
python3 main.py \
  --provider modal-vllm \
  --endpoint https://your-workspace--example-vllm-inference-serve.modal.run \
  --model llm \
  --api-key EMPTY \
  --dataset simp \
  --limit 1 \
  --max-tokens 512 \
  --output outputs/predictions.jsonl
```

Use the real OpenAI API instead:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini

python3 main.py \
  --provider openai \
  --dataset simp \
  --limit 1 \
  --max-tokens 512 \
  --output outputs/predictions_openai.jsonl
```

Generate the complex dataset:

```bash
python3 main.py --provider modal-vllm --dataset comp --limit 1
```

The generated file is written to `outputs/predictions.jsonl` by default.

## Scope

In scope:

- read `data/TritonBench_T_<simp|comp>_alpac_v1.json`
- build prompt messages
- call either `modal_vllm` or OpenAI through the `openai` client
- write `predictions.jsonl`

Out of scope:

- local Hugging Face inference
- XGrammar runtime execution
- Modal image build
- TritonBench eval script patching
- benchmark execution
- uploading predictions
