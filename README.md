# LLM Pipeline Playground

This project packages a local Hugging Face chat interface, OpenAI chat with tracing, activation inspection, and supervised fine-tuning with MLflow tracking. Everything runs with one `docker compose up`.

## Prerequisites
- Docker + Docker Compose
- OpenAI API key (set `OPENAI_API_KEY`)

## Quickstart
1. Set environment variables (optional overrides shown):
   ```bash
   export OPENAI_API_KEY=sk-...
   export HF_MODEL_NAME=gpt2
   export OPENAI_MODEL_NAME=gpt-4o-mini
   ```
2. Build and launch:
   ```bash
   docker compose build
   docker compose up
   ```
3. Access services:
   - App UI: http://localhost:7860
   - MLflow UI: http://localhost:5000
   - Minio console: http://localhost:9000 (user/pass `minioadmin`)

### Create the MLflow bucket in Minio
If the `mlflow` bucket does not exist, create it via the Minio console or `mc`:
```bash
mc alias set local http://localhost:9000 minioadmin minioadmin
mc mb local/mlflow
```

## How it works
- MLflow is configured at startup and `mlflow.openai.autolog()` is enabled before any OpenAI calls, so every OpenAI request (Responses API by default, Chat Completions optional) is traced with prompts and responses.
- Postgres stores MLflow metadata; Minio stores artifacts and models.
- The app automatically selects CUDA if available, else CPU.

## Gradio tabs
- **Local HF Chat**: Chats with the local HF causal LM (default `gpt2`).
- **OpenAI Chat**: Uses OpenAI Responses API by default; toggle to Chat Completions if desired. All calls are traced in MLflow.
- **Activation Explorer**: Select layers, run a prompt, view stats and a plot, and optionally log activations/plots to MLflow.
- **Fine tuning**: Launch supervised fine-tuning (JSONL with `instruction`/`output`) as a background process and get the PID/command. View results and artifacts in the MLflow UI.

## Dataset format for fine-tuning
JSONL where each line is `{"instruction": "...", "output": "..."}`. Paths are read from inside the container (mount or bake data accordingly).

## Environment variables
Key settings (defaults for Docker):
- `OPENAI_API_KEY` (required for OpenAI tab)
- `HF_MODEL_NAME` (default `gpt2`)
- `OPENAI_MODEL_NAME` (default `gpt-4o-mini`)
- `USE_RESPONSES_API` (default `True`)
- `USE_VLLM` (default `False`, requires GPU; when `True` and CUDA is available, local chat routes to a vLLM OpenAI-compatible endpoint)
- `VLLM_SERVER_URL` (default `http://vllm:8000/v1`)
- `VLLM_MODEL_NAME` (optional override for vLLM served model)
- `MLFLOW_TRACKING_URI` (default `http://mlflow:5000`)
- `MLFLOW_S3_ENDPOINT_URL` (default `http://minio:9000`)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (default `minioadmin`/`minioadmin`)

### Optional vLLM backend
- The compose file includes a commented `vllm` service skeleton for GPU hosts. Uncomment and adjust model paths to serve via vLLM.
- Set `USE_VLLM=True` (and `VLLM_SERVER_URL` if changed) to route Local HF Chat through vLLM; falls back to in-process HF when disabled or GPU is unavailable.
