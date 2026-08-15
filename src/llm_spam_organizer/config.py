"""Configuration loading from a TOML file."""

from __future__ import annotations

import os
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .prompts import PROMPTS


class ConfigError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ImapConfig:
    host: str
    username: str
    password: str
    port: int = 993
    inbox: str = "INBOX"
    spam_folder: str = "Junk"
    starttls: bool = False


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    model: str = "qwen3.5:2b-q4_K_M"
    host: str = "http://localhost:11434"
    prompt: str = "concise"
    threshold: float = 0.5
    max_body_chars: int = 4000
    num_ctx: int = 4096
    temperature: float = 0.0
    # Two parallel requests already saturate a laptop CPU and keep the KV cache small.
    concurrency: int = 2
    # Reasoning traces cost minutes per message on CPU without improving the verdict.
    think: bool = False
    keep_alive: str = "5m"
    timeout: float = 600.0


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    archive_folders: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    ham_sample: int = 200
    spam_sample: int = 200
    seed: int = 1234
    output_dir: Path = Path("evaluation")


@dataclass(frozen=True, slots=True)
class Config:
    imap: ImapConfig
    ollama: OllamaConfig
    evaluation: EvaluationConfig
    state_path: Path
    source: Path


def default_config_path() -> Path:
    return _xdg_dir("XDG_CONFIG_HOME", ".config") / "config.toml"


def default_state_path() -> Path:
    return _xdg_dir("XDG_STATE_HOME", ".local/state") / "state.json"


def _xdg_dir(variable: str, fallback: str) -> Path:
    base = os.environ.get(variable) or Path.home() / fallback
    return Path(base) / "llm-spam-organizer"


def load_config(path: Path) -> Config:
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(
            f"No configuration at {path}, create one with `llm-spam-organizer init-config`"
        ) from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path} is not valid TOML: {error}") from None
    return parse_config(raw, source=path)


def parse_config(raw: dict[str, Any], *, source: Path = Path("<memory>")) -> Config:
    unknown = raw.keys() - {"imap", "ollama", "evaluation", "state_path"}
    if unknown:
        raise ConfigError(f"Unknown top level sections: {sorted(unknown)}")

    imap = dict(raw.get("imap", {}))
    if "password_command" in imap:
        if "password" in imap:
            raise ConfigError("Set either `password` or `password_command`, not both")
        imap["password"] = _run_password_command(imap.pop("password_command"))
    evaluation = dict(raw.get("evaluation", {}))
    if "output_dir" in evaluation:
        evaluation["output_dir"] = Path(evaluation["output_dir"]).expanduser()

    config = Config(
        imap=_section(ImapConfig, imap, "imap"),
        ollama=_section(OllamaConfig, raw.get("ollama", {}), "ollama"),
        evaluation=_section(EvaluationConfig, evaluation, "evaluation"),
        state_path=Path(raw.get("state_path", default_state_path())).expanduser(),
        source=source,
    )
    if not 0.0 <= config.ollama.threshold <= 1.0:
        raise ConfigError("ollama.threshold must be within [0, 1]")
    if config.ollama.concurrency < 1:
        raise ConfigError("ollama.concurrency must be at least 1")
    unknown_prompts = {config.ollama.prompt, *config.evaluation.prompts} - set(PROMPTS)
    if unknown_prompts:
        raise ConfigError(
            f"Unknown prompt(s) {sorted(unknown_prompts)}, available: {sorted(PROMPTS)}"
        )
    return config


def _section[T](cls: type[T], values: dict[str, Any], name: str) -> T:
    known = {f.name for f in fields(cls)}
    unknown = values.keys() - known
    if unknown:
        raise ConfigError(f"Unknown keys in [{name}]: {sorted(unknown)}")
    try:
        return cls(**values)
    except TypeError as error:
        raise ConfigError(f"Incomplete [{name}] section: {error}") from None


def _run_password_command(command: str | list[str]) -> str:
    argv = shlex.split(command) if isinstance(command, str) else command
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConfigError(f"password_command failed: {error}") from None
    password = result.stdout.split("\n", 1)[0].strip()
    if not password:
        raise ConfigError("password_command produced no output")
    return password


EXAMPLE_CONFIG = """\
[imap]
host = "imap.example.com"
port = 993
username = "me@example.com"
# Either a literal password or a command that prints it on the first line.
password_command = "pass show mail/example.com"
inbox = "INBOX"
spam_folder = "Junk"

[ollama]
host = "http://localhost:11434"
model = "qwen3.5:2b-q4_K_M"
prompt = "concise"
# Messages scoring at or above this spam probability are moved.
threshold = 0.7
max_body_chars = 4000
num_ctx = 4096
# More than a few parallel requests only grows the KV cache on a laptop.
concurrency = 2
# A reasoning trace costs minutes per message on CPU and did not improve the verdicts.
think = false

[evaluation]
archive_folders = ["Archive", "Archive/2025"]
models = ["qwen3.5:0.8b", "qwen3.5:2b-q4_K_M", "qwen3.5:4b-q4_K_M"]
prompts = ["concise", "detailed", "rubric"]
ham_sample = 200
spam_sample = 200
seed = 1234
output_dir = "evaluation"
"""
