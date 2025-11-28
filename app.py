import logging
import os
from typing import Dict, List, Optional

import gradio as gr
import psutil

import config
from models import hf_model, openai_client, router as model_router
from monitoring import activations_viz, mlflow_utils
from monitoring.tracing_openai import setup_tracing
from training import launcher
from huggingface_hub import snapshot_download, login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure MLflow and OpenAI tracing at startup
setup_tracing()
router = model_router.get_default_router()


def _ensure_model_cached():
    """Download the HF model snapshot into /models_cache for vLLM or local use."""
    model_id = config.VLLM_MODEL_NAME or config.HF_MODEL_NAME
    token = getattr(config, "HUGGINGFACE_HUB_TOKEN", None)
    target_dir = os.path.join("/models_cache", model_id.replace("/", "__"))
    os.makedirs(target_dir, exist_ok=True)
    try:
        if token:
            login(token=token)
        snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            revision="main",
        )
        logger.info("Model %s cached at %s", model_id, target_dir)
    except Exception as exc:
        logger.warning("Could not cache model %s: %s", model_id, exc)


_ensure_model_cached()


def _ensure_history(history: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
    return list(history) if history else []


def _history_with_user(history: List[Dict[str, str]], user_message: str) -> List[Dict[str, str]]:
    new_hist = list(history)
    new_hist.append({"role": "user", "content": user_message or ""})
    return new_hist


def _to_plain(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    plain: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        plain.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return plain


def hf_chat_fn(user_message: str, history: List[Dict[str, str]] | None):
    history = _ensure_history(history)
    messages = _history_with_user(history, user_message)
    try:
        response = router.generate_local_response(_to_plain(messages))
    except Exception as exc:
        logger.error("HF chat failed: %s", exc)
        response = f"[vLLM error: {exc}] Ensure vLLM is running and USE_VLLM=True."
    messages.append({"role": "assistant", "content": str(response)})
    status = f"Backend: vLLM (GPU only) | target model: {config.VLLM_MODEL_NAME or 'unset'}"
    return messages, status


def openai_chat_fn(user_message: str, history: List[Dict[str, str]] | None, use_responses: bool = True):
    history = _ensure_history(history)
    messages = _history_with_user(history, user_message)
    reply = openai_client.chat_completion(_to_plain(messages), use_responses=use_responses)
    messages.append({"role": "assistant", "content": reply})
    return messages


def _get_layer_choices() -> List[str]:
    # Avoid loading model on startup; provide sensible defaults
    return ["transformer.h.0", "transformer.h.1", "transformer"]


def activation_fn(text: str, layers: List[str], log_to_mlflow: bool = False):
    if not text:
        return "Please provide text to inspect.", None
    try:
        model, tokenizer, _ = hf_model.load_model()
        activations = activations_viz.compute_activations(model, tokenizer, text, layers)
    except Exception as exc:
        logger.error("Activation computation failed: %s", exc)
        return f"[HF error: {exc}] Check HF_MODEL_NAME access/token.", None
    if not activations:
        return "No activations captured; check selected layers.", None

    stats_lines = []
    for name, arr in activations.items():
        stats_lines.append(
            f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}"
        )

    if log_to_mlflow:
        activations_viz.log_activations_to_mlflow(activations)

    figs = activations_viz.plot_activation_summary(activations)
    first_fig = next(iter(figs.values())) if figs else None
    return "\n".join(stats_lines), first_fig


MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", f"{config.MLFLOW_TRACKING_URI}")


def start_finetune(base_model: str, train_path: str, eval_path: str, epochs: float, batch_size: int, lr: float, output_dir: str):
    params = {
        "model_name": base_model or config.HF_MODEL_NAME,
        "train_path": train_path,
        "eval_path": eval_path or None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "output_dir": output_dir or "./outputs",
    }
    job = launcher.launch_finetune_subprocess(params)
    link = f"MLflow UI: {MLFLOW_UI_URL}"
    summary = f"Started job {job['job_id']} (PID {job['pid']})\nCommand: {job['command']}\n{link}"
    return summary, str(job["pid"])


def refresh_status(pid: str):
    try:
        pid_int = int(pid)
    except ValueError:
        return "Invalid PID"
    if psutil.pid_exists(pid_int):
        proc = psutil.Process(pid_int)
        if proc.is_running():
            return f"Process {pid_int} is running (status: {proc.status()})"
    return f"Process {pid_int} not running"


def build_interface() -> gr.Blocks:
    layer_choices = _get_layer_choices()
    with gr.Blocks(title="LLM Pipeline") as demo:
        gr.Markdown("## LLM Playground with HF + OpenAI + MLflow")

        with gr.Tab("Local HF Chat"):
            backend_label = f"Backend: vLLM (GPU only). Ensure USE_VLLM=True and VLLM server reachable at {config.VLLM_SERVER_URL}"
            hf_status = gr.Markdown(backend_label)
            hf_chatbot = gr.Chatbot(height=400, type="messages")
            hf_msg = gr.Textbox(label="Message")
            hf_send = gr.Button("Send")
            hf_clear = gr.Button("Clear")

            hf_send.click(hf_chat_fn, inputs=[hf_msg, hf_chatbot], outputs=[hf_chatbot, hf_status])
            hf_msg.submit(hf_chat_fn, inputs=[hf_msg, hf_chatbot], outputs=[hf_chatbot, hf_status])
            hf_clear.click(lambda: [], None, hf_chatbot, queue=False)

        with gr.Tab("OpenAI Chat"):
            oa_chatbot = gr.Chatbot(height=400)
            oa_msg = gr.Textbox(label="Message")
            use_resp = gr.Checkbox(value=True, label="Use Responses API")
            oa_send = gr.Button("Send")
            oa_clear = gr.Button("Clear")

            oa_send.click(openai_chat_fn, inputs=[oa_msg, oa_chatbot, use_resp], outputs=oa_chatbot)
            oa_msg.submit(openai_chat_fn, inputs=[oa_msg, oa_chatbot, use_resp], outputs=oa_chatbot)
            oa_clear.click(lambda: [], None, oa_chatbot, queue=False)

        with gr.Tab("Activation Explorer"):
            prompt = gr.Textbox(label="Text", lines=3)
            layer_select = gr.CheckboxGroup(layer_choices, value=layer_choices[:2], label="Layers")
            log_toggle = gr.Checkbox(value=False, label="Log to MLflow")
            act_button = gr.Button("Compute")
            stats = gr.Markdown()
            fig_plot = gr.Plot(label="Activation Plot")

            act_button.click(activation_fn, inputs=[prompt, layer_select, log_toggle], outputs=[stats, fig_plot])

        with gr.Tab("Fine tuning"):
            base_model = gr.Textbox(label="Base HF model", value=config.HF_MODEL_NAME)
            train_path = gr.Textbox(label="Training JSONL path")
            eval_path = gr.Textbox(label="Eval JSONL path (optional)")
            epochs = gr.Number(label="Epochs", value=1.0)
            batch_size = gr.Number(label="Batch size", value=2, precision=0)
            lr = gr.Number(label="Learning rate", value=5e-5)
            output_dir = gr.Textbox(label="Output dir", value="./outputs")
            start_btn = gr.Button("Start fine tuning")
            refresh_btn = gr.Button("Refresh status")
            pid_box = gr.Textbox(label="PID", interactive=False)
            status_box = gr.Textbox(label="Status")

            start_btn.click(
                start_finetune,
                inputs=[base_model, train_path, eval_path, epochs, batch_size, lr, output_dir],
                outputs=[status_box, pid_box],
            )
            refresh_btn.click(refresh_status, inputs=[pid_box], outputs=[status_box])

    return demo


def main():
    demo = build_interface()
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
