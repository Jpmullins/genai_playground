# GenAI Playground

GenAI Playground is a Python toolkit for fine-tuning, evaluating, and serving large language models via CLI, Web UI, or API.

## Requirements
- Python 3.9+
- Linux/macOS/WSL; CUDA/NPU GPUs recommended for training, CPU fallback for light tests
- Git, build tools, and recent `pip`

## Installation
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
# add extras as needed: pip install -e .[torch] .[vllm] ...
```

## Quickstart (CLI)
Train with a provided config (uses demo datasets/models):
```bash
genaiplayground-cli train examples/train_lora/llama3_lora_sft.yaml
# or with GPU selection
CUDA_VISIBLE_DEVICES=0 genaiplayground-cli train examples/train_lora/llama3_lora_sft.yaml
```
- Short alias: `gap train ...`
- To avoid running arbitrary model code, keep `trust_remote_code` disabled unless you trust the source.

Run inference from the CLI:
```bash
gap chat examples/inference/llama3_lora_sft.yaml
```

## Web UI
Launch the Gradio interface (opens browser; set `GRADIO_SHARE=true` to tunnel, `API_KEY` for auth):
```bash
gap webui
```

## API Server
OpenAI-compatible FastAPI server (set `API_KEY` to enforce auth):
```bash
gap api
# visit http://localhost:8000/docs
```

## Docker
- CUDA compose (from repo root, requires NVIDIA runtime):
```bash
cd docker/docker-cuda
docker compose up -d
docker compose exec genaiplayground bash
# inside container
gap webui  # or gap train ...
```
- Build/run manually with GPUs:
```bash
docker build -t genaiplayground:latest -f docker/docker-cuda/Dockerfile .
docker run -dit --ipc=host --gpus=all --name genaiplayground \
  -v $PWD:/workspace -w /workspace genaiplayground:latest bash
docker exec -it genaiplayground bash
```
- NPU/ROCm variants: see `docker/docker-npu` and `docker/docker-rocm`.

## Testing & Quality
```bash
make quality   # Ruff lint/format check
make style     # Auto-fix formatting
make test      # Pytest suites (CPU-only by default)
```

## Project Layout
- `src/genaiplayground`: core library (data, models, training, chat, webui, API)
- `examples/`: ready-to-run training/inference configs (LoRA, QLoRA, full fine-tuning, eval/export)
- `scripts/`: utility scripts (metrics, conversions, benchmarking)
- `tests/` and `tests_v1/`: current and legacy test suites
- `docker/`: container recipes
- `data/`: demo datasets, metadata, and tiny fixtures used in tests/examples

## Data & Examples
- Demo datasets are described in `data/dataset_info.json`; use `dataset=genaiplayground/<name>` in YAML configs to pull tiny, fast-running sets (e.g., `alpaca_en_demo`, `glaive_toolcall_en`).
- Environment helpers: `DEMO_DATA` and `TINY_LLAMA3`/`TINY_DATA` control default paths/models for tests and examples.
- Training configs live under `examples/` (e.g., `examples/train_lora/llama3_lora_sft.yaml`); run them directly with `gap train <config>` and override params inline (`learning_rate=... logging_steps=...`).
- Inference/export examples are under `examples/inference` and `examples/merge_lora`; they align with the training configs so you can fine-tune, then chat or merge weights without rewriting args.

## Hugging Face Token
- Set `HUGGINGFACE_HUB_TOKEN=hf_xxx` in your shell (or in `docker-compose.yml` env) before pulling private models/datasets:
  ```bash
  export HUGGINGFACE_HUB_TOKEN=hf_your_token_here
  gap train examples/train_lora/llama3_lora_sft.yaml
  ```
- Alternatively, run `huggingface-cli login` (inside the container or host) to cache the token under your home directory.

## Tips
- Use small demo datasets (`data/dataset_info.json`) and env vars like `TINY_LLAMA3`, `DEMO_DATA` to keep runs fast.
- Disable telemetry with `WANDB_DISABLED=true`; set `HF_HUB_OFFLINE=1` for offline mode if artifacts are pre-cached.
- When exposing the API/UI, enable `API_KEY`, restrict hosts, and front with TLS/rate limiting.
