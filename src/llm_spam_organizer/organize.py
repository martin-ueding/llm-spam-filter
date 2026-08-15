"""Move spam from the inbox into the spam folder."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from imap_tools import ImapToolsError, MailBox

from .classify import ClassificationError, SpamClassifier, Verdict
from .config import Config
from .mail import Mail, fetch_mails, fetch_uids_above, open_mailbox, uid_validity

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Run:
    seen: int = 0
    moved: int = 0
    failed: int = 0

    def add(self, other: Run) -> None:
        self.seen += other.seen
        self.moved += other.moved
        self.failed += other.failed


def organize(
    config: Config,
    *,
    dry_run: bool = False,
    reprocess: bool = False,
    limit: int | None = None,
) -> Run:
    classifier = SpamClassifier(config.ollama)
    state = State(config.state_path)
    with open_mailbox(config.imap) as mailbox:
        key = _state_key(config, mailbox)
        return _process(
            mailbox,
            config,
            classifier,
            state,
            key,
            dry_run=dry_run,
            reprocess=reprocess,
            limit=limit,
        )


def watch(
    config: Config, *, dry_run: bool = False, timeout: float = 600, retry_delay: float = 30
) -> None:
    """Process new mail as it arrives, using IMAP IDLE, reconnecting when the link drops."""
    classifier = SpamClassifier(config.ollama)
    state = State(config.state_path)
    while True:
        try:
            with open_mailbox(config.imap) as mailbox:
                key = _state_key(config, mailbox)
                while True:
                    _process(mailbox, config, classifier, state, key, dry_run=dry_run)
                    logger.debug("Waiting for new mail in %s", config.imap.inbox)
                    mailbox.idle.wait(timeout=timeout)
        except (ImapToolsError, OSError) as error:
            logger.warning("Connection lost (%s), reconnecting in %.0f s", error, retry_delay)
            time.sleep(retry_delay)


def _process(
    mailbox: MailBox,
    config: Config,
    classifier: SpamClassifier,
    state: State,
    key: str,
    *,
    dry_run: bool,
    reprocess: bool = False,
    limit: int | None = None,
) -> Run:
    last_uid = 0 if reprocess else state.get(key)
    uids = fetch_uids_above(mailbox, last_uid)
    if limit is not None:
        uids = uids[-limit:]
    run = Run()
    if not uids:
        logger.info("No messages above UID %d in %s", last_uid, config.imap.inbox)
        return run

    logger.info("Classifying %d message(s) with %s", len(uids), classifier.model)
    mails = list(fetch_mails(mailbox, config.imap.inbox, uids))
    spam_uids: list[str] = []
    for mail, result in classifier.classify_many(mails):
        run.seen += 1
        if isinstance(result, ClassificationError):
            run.failed += 1
            logger.warning("UID %s could not be classified: %s", mail.uid, result)
            continue
        if result.is_spam(config.ollama.threshold):
            spam_uids.append(mail.uid)
        logger.info("%s", _summary(mail, result, config.ollama.threshold))

    run.moved = len(spam_uids)
    if dry_run:
        logger.info("Dry run: would move %d message(s)", run.moved)
        return run

    if spam_uids:
        mailbox.move(spam_uids, config.imap.spam_folder)
        logger.info("Moved %d message(s) to %s", run.moved, config.imap.spam_folder)
    state.set(key, max(int(uid) for uid in uids))
    return run


def _state_key(config: Config, mailbox: MailBox) -> str:
    validity = uid_validity(mailbox, config.imap.inbox)
    return f"{config.imap.username}@{config.imap.host}/{config.imap.inbox}#{validity}"


def _summary(mail: Mail, verdict: Verdict, threshold: float) -> str:
    label = "SPAM" if verdict.is_spam(threshold) else "ham "
    subject = mail.subject[:60] or "<no subject>"
    return f"{label} p={verdict.spam_probability:.2f} {mail.sender} | {subject} | {verdict.reason}"


class State:
    """Highest already classified UID per account, folder, and UIDVALIDITY."""

    def __init__(self, path: Path) -> None:
        self.path = path
        try:
            self._data: dict[str, int] = json.loads(path.read_text())
        except FileNotFoundError, json.JSONDecodeError:
            self._data = {}

    def get(self, key: str) -> int:
        return int(self._data.get(key, 0))

    def set(self, key: str, uid: int) -> None:
        if uid <= self.get(key):
            return
        self._data[key] = uid
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        temporary.replace(self.path)
