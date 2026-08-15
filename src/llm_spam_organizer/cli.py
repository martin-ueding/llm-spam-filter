"""Command line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from imap_tools import ImapToolsError
from imap_tools.message import MailMessage

from . import evaluate as evaluation_module
from . import organize as organize_module
from .classify import ClassificationError, SpamClassifier
from .config import EXAMPLE_CONFIG, Config, ConfigError, default_config_path, load_config
from .mail import Mail
from .prompts import PROMPTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-spam-organizer",
        description="Move spam into the spam folder using a local LLM served by Ollama.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path(),
        help="TOML configuration file (default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="log debug messages")
    parser.add_argument("-q", "--quiet", action="store_true", help="log warnings only")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="classify new inbox mail and move the spam")
    run.add_argument("-n", "--dry-run", action="store_true", help="classify but do not move")
    run.add_argument(
        "--reprocess", action="store_true", help="ignore the stored UID and start over"
    )
    run.add_argument("--limit", type=int, help="only look at the newest N messages")
    run.add_argument(
        "--watch", action="store_true", help="keep running and wait for new mail via IMAP IDLE"
    )
    run.set_defaults(function=_run)

    evaluate = subparsers.add_parser(
        "evaluate", help="compare models and prompts on the spam and archive folders"
    )
    evaluate.add_argument(
        "--refresh", action="store_true", help="fetch the dataset again instead of using the cache"
    )
    evaluate.add_argument(
        "--rescore", action="store_true", help="discard cached scores and classify again"
    )
    evaluate.set_defaults(function=_evaluate)

    classify = subparsers.add_parser("classify", help="classify RFC 822 messages from files")
    classify.add_argument("paths", nargs="*", type=Path, help="*.eml files, or stdin if omitted")
    classify.add_argument("--model", help="override the configured model")
    classify.add_argument(
        "--prompt", choices=sorted(PROMPTS), help="override the configured prompt"
    )
    classify.add_argument("--show-input", action="store_true", help="print what the model sees")
    classify.set_defaults(function=_classify)

    init = subparsers.add_parser("init-config", help="write an example configuration file")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(function=_init_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    level = (
        logging.DEBUG if arguments.verbose else logging.WARNING if arguments.quiet else logging.INFO
    )
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(message)s")

    try:
        return arguments.function(arguments)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (ImapToolsError, OSError) as error:
        print(f"IMAP error: {error}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130


def _config(arguments: argparse.Namespace) -> Config:
    return load_config(arguments.config)


def _run(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    if arguments.watch:
        organize_module.watch(config, dry_run=arguments.dry_run)
        return 0
    run = organize_module.organize(
        config,
        dry_run=arguments.dry_run,
        reprocess=arguments.reprocess,
        limit=arguments.limit,
    )
    verb = "would move" if arguments.dry_run else "moved"
    print(f"Classified {run.seen}, {verb} {run.moved}, failed {run.failed}")
    return 1 if run.failed and not run.seen else 0


def _evaluate(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    curves = evaluation_module.evaluate(
        config, refresh=arguments.refresh, rescore=arguments.rescore
    )
    if not curves:
        print("No ROC curve could be computed; the dataset needs both spam and ham.")
        return 1
    print()
    print(evaluation_module.format_table(curves))
    return 0


def _classify(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    classifier = SpamClassifier(config.ollama, model=arguments.model, prompt=arguments.prompt)
    mails = [_read_eml(path) for path in arguments.paths] or [_read_eml(None)]
    for mail, result in classifier.classify_many(mails):
        if arguments.show_input:
            print(mail.render(config.ollama.max_body_chars))
            print("-" * 72)
        if isinstance(result, ClassificationError):
            print(f"error: {result}", file=sys.stderr)
            return 1
        label = "spam" if result.is_spam(config.ollama.threshold) else "ham"
        print(f"{label} p={result.spam_probability:.2f} {mail.subject!r}: {result.reason}")
    return 0


def _read_eml(path: Path | None) -> Mail:
    raw = path.read_bytes() if path else sys.stdin.buffer.read()
    return Mail.from_message(MailMessage.from_bytes(raw), str(path or "<stdin>"))


def _init_config(arguments: argparse.Namespace) -> int:
    path = arguments.config
    if path.exists() and not arguments.force:
        print(f"{path} already exists, pass --force to overwrite", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG)
    path.chmod(0o600)
    print(f"Wrote {path}")
    return 0
