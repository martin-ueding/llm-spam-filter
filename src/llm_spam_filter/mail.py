"""IMAP access and extraction of the text an LLM gets to see."""

from __future__ import annotations

import re
import ssl
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Self

import html2text
from imap_tools import AND, MailBox, MailBoxStartTls, MailMessage

from .config import ImapConfig

_INTERESTING_HEADERS = ("List-Unsubscribe", "Return-Path", "X-Mailer", "Precedence")
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Mail:
    """A message reduced to what is worth showing to the classifier."""

    uid: str
    folder: str
    sender: str
    recipients: str
    subject: str
    date: str
    body: str
    attachments: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_message(cls, message: MailMessage, folder: str) -> Self:
        return cls(
            uid=message.uid or "",
            folder=folder,
            sender=message.from_values.full if message.from_values else message.from_ or "",
            recipients=", ".join(address.full for address in message.to_values),
            subject=message.subject or "",
            date=message.date_str or "",
            body=_body_text(message),
            attachments=[a.filename or "<unnamed>" for a in message.attachments],
            headers={
                name: values[0]
                for name in _INTERESTING_HEADERS
                if (values := message.headers.get(name.lower()))
            },
        )

    def render(self, max_body_chars: int) -> str:
        """Format the message as plain text for the prompt."""
        lines = [
            f"From: {self.sender}",
            f"To: {self.recipients}",
            f"Subject: {self.subject}",
            f"Date: {self.date}",
        ]
        lines += [f"{name}: {value}" for name, value in self.headers.items()]
        if self.attachments:
            lines.append(f"Attachments: {', '.join(self.attachments)}")
        body = self.body
        if len(body) > max_body_chars:
            body = body[:max_body_chars] + "\n[truncated]"
        return "\n".join(lines) + "\n\n" + body

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(**data)


def _body_text(message: MailMessage) -> str:
    text = message.text or ""
    if not text.strip() and message.html:
        converter = html2text.HTML2Text()
        converter.ignore_images = True
        converter.ignore_tables = False
        converter.body_width = 0
        text = converter.handle(message.html)
    return _collapse(text)


def _collapse(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _TRAILING_SPACE.sub("", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


@contextmanager
def open_mailbox(config: ImapConfig, folder: str | None = None) -> Iterator[MailBox]:
    box_class = MailBoxStartTls if config.starttls else MailBox
    box = box_class(config.host, port=config.port, ssl_context=ssl.create_default_context())
    initial_folder = folder or config.inbox
    with box.login(config.username, config.password, initial_folder=initial_folder) as mailbox:
        yield mailbox


def uid_validity(mailbox: MailBox, folder: str) -> str:
    """UIDs are only comparable within one UIDVALIDITY generation."""
    status = mailbox.folder.status(folder, ["UIDVALIDITY"])
    return str(status.get("UIDVALIDITY", "0"))


def fetch_uids_above(mailbox: MailBox, last_uid: int) -> list[str]:
    return sorted(
        (uid for uid in mailbox.uids() if uid.isdigit() and int(uid) > last_uid),
        key=int,
    )


def fetch_mails(mailbox: MailBox, folder: str, uids: list[str]) -> Iterator[Mail]:
    if not uids:
        return
    for message in mailbox.fetch(AND(uid=uids), mark_seen=False, bulk=50):
        yield Mail.from_message(message, folder)
