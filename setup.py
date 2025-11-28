import os
import re
from pathlib import Path

from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("src", "genaiplayground", "extras", "env.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_console_scripts() -> list[str]:
    console_scripts = ["genaiplayground-cli = genaiplayground.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("gap = genaiplayground.cli:main")

    return console_scripts


extra_require = {
    "torch": ["torch>=2.0.0", "torchvision>=0.15.0"],
    "torch-npu": ["torch==2.7.1", "torch-npu==2.7.1", "torchvision==0.22.1", "decorator"],
    "metrics": ["nltk", "jieba", "rouge-chinese"],
    "deepspeed": ["deepspeed>=0.10.0,<=0.16.9"],
    "liger-kernel": ["liger-kernel>=0.5.5"],
    "bitsandbytes": ["bitsandbytes>=0.39.0"],
    "hqq": ["hqq"],
    "eetq": ["eetq"],
    "gptq": ["optimum>=1.24.0", "gptqmodel>=2.0.0"],
    "aqlm": ["aqlm[gpu]>=1.1.0"],
    "vllm": ["vllm>=0.4.3,<=0.11.0"],
    "sglang": ["sglang[srt]>=0.4.5", "transformers==4.51.1"],
    "galore": ["galore-torch"],
    "apollo": ["apollo-torch"],
    "badam": ["badam>=1.2.1"],
    "adam-mini": ["adam-mini"],
    "minicpm_v": [
        "soundfile",
        "torchvision",
        "torchaudio",
        "vector_quantize_pytorch",
        "vocos",
        "msgpack",
        "referencing",
        "jsonschema_specifications",
    ],
    "openmind": ["openmind"],
    "swanlab": ["swanlab"],
    "fp8": ["torchao>=0.8.0", "accelerate>=1.10.0"],
    "fp8-te": ["transformer_engine[pytorch]>=2.0.0", "accelerate>=1.10.0"],
    "fp8-all": ["torchao>=0.8.0", "transformer_engine[pytorch]>=2.0.0", "accelerate>=1.10.0"],
    "dev": ["pre-commit", "ruff", "pytest", "build"],
}


def main():
    try:
        long_description = Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        long_description = "GenAI Playground: tools for experimenting with LLM fine-tuning, evaluation, and serving."

    setup(
        name="genai-playground",
        version=get_version(),
        author="hiyouga",
        author_email="hiyouga@buaa.edu.cn",
        description="GenAI Playground for training and serving language models.",
        long_description=long_description,
        long_description_content_type="text/markdown",
        keywords=["AI", "LLM", "GPT", "ChatGPT", "Llama", "Transformer", "DeepSeek", "Pytorch"],
        url="https://github.com/genai-playground/genai-playground",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
