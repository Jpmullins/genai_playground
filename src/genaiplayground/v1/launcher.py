import sys

from ..extras.env import VERSION, print_env


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   genaiplayground-cli sft -h: train models                         |\n"
    + "|   genaiplayground-cli version: show version info                   |\n"
    + "| Hint: You can use `gap` as a shortcut for `genaiplayground-cli`.   |\n"
    + "-" * 70
)


WELCOME = (
    "-" * 58
    + "\n"
    + f"| Welcome to GenAI Playground, version {VERSION}"
    + " " * (21 - len(VERSION))
    + "|\n|"
    + " " * 56
    + "|\n"
    + "| Project page: https://github.com/genai-playground/genai-playground |\n"
    + "-" * 58
)


def launch():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else "help"

    if command == "sft":
        from .trainers.sft_trainer import run_sft

        run_sft()

    elif command == "env":
        print_env()

    elif command == "version":
        print(WELCOME)

    elif command == "help":
        print(USAGE)

    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    pass
