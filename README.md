# deepbork

Local prediction-generation orchestrator for Deepbork.

This repo only creates a `predictions.jsonl` file. It does not run local model
inference, build a Modal image, execute TritonBench evaluation, or upload
artifacts anywhere.

The current flow is:

1. read the local TritonBench-T Alpaca dataset files from `data/`
2. build OpenAI-style chat messages for each benchmark item
3. call a deployed OpenAI-compatible LLM endpoint, such as `modal_vllm`
4. clean the returned code block
5. write one JSON line per generated operator

## Contract

The LLM endpoint receives OpenAI-compatible chat completion requests:

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
```

No full TritonBench clone is needed for generation.

## Setup

Install the lightweight orchestrator dependency:

```bash
python3 -m pip install -r requirements.txt
```

Create a `.env` file or export the endpoint in your shell:

```bash
DEFAULT_ENDPOINT=https://your-workspace--example-vllm-inference-serve.modal.run
DEFAULT_MODEL=llm
```

The endpoint should be the base URL from `modal_vllm`, without
`/v1/chat/completions`.

## Generate Predictions

Smoke test one item:

```bash
python3 main.py --limit 1
```

Pass the endpoint explicitly:

```bash
python3 main.py \
  --endpoint https://your-workspace--example-vllm-inference-serve.modal.run \
  --model llm \
  --dataset simp \
  --limit 1 \
  --max-tokens 512 \
  --output outputs/predictions.jsonl
```

Generate the complex dataset:

```bash
python3 main.py --dataset comp --limit 1
```

The generated file is written to `outputs/predictions.jsonl` by default.

## Scope

In scope:

- read `data/TritonBench_T_<simp|comp>_alpac_v1.json`
- build prompt messages
- call an OpenAI-compatible chat completion endpoint
- write `predictions.jsonl`

Out of scope:

- local Hugging Face inference
- XGrammar runtime execution
- Modal image build
- TritonBench eval script patching
- benchmark execution
- uploading predictions
