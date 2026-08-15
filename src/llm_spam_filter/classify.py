"""Spam classification through a local LLM served by Ollama."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import ollama
from pydantic import BaseModel, Field

from .config import OllamaConfig
from .mail import Mail
from .prompts import get_prompt

logger = logging.getLogger(__name__)


class ClassificationError(Exception):
    pass


class _Response(BaseModel):
    """Schema handed to Ollama as structured output; the reason comes first on purpose."""

    reason: str = Field(description="One sentence justifying the score.")
    spam_probability: float = Field(ge=0.0, le=1.0)


@dataclass(frozen=True, slots=True)
class Verdict:
    spam_probability: float
    reason: str
    model: str
    prompt: str

    def is_spam(self, threshold: float) -> bool:
        return self.spam_probability >= threshold


class SpamClassifier:
    def __init__(
        self,
        config: OllamaConfig,
        *,
        model: str | None = None,
        prompt: str | None = None,
    ) -> None:
        self.config = config
        self.model = model or config.model
        self.prompt_name = prompt or config.prompt
        self.system_prompt = get_prompt(self.prompt_name)
        self._client = ollama.Client(host=config.host, timeout=config.timeout)
        self._think: bool | None = config.think

    def classify(self, mail: Mail) -> Verdict:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": mail.render(self.config.max_body_chars)},
        ]
        try:
            response = self._chat(messages)
        except (ollama.ResponseError, ConnectionError, TimeoutError) as error:
            raise ClassificationError(f"{self.model}: {error}") from None

        self._check_context(response, mail)
        content = response.message.content or ""
        try:
            parsed = _Response.model_validate_json(content)
        except ValueError as error:
            raise ClassificationError(
                f"{self.model} returned no usable verdict: {content[:200]!r} ({error})"
            ) from None
        return Verdict(
            spam_probability=parsed.spam_probability,
            reason=parsed.reason.strip(),
            model=self.model,
            prompt=self.prompt_name,
        )

    def _check_context(self, response: ollama.ChatResponse, mail: Mail) -> None:
        """An overlong prompt is shifted out at the front, taking the system prompt with it."""
        used = response.prompt_eval_count or 0
        limit = self.config.num_ctx
        if limit and used > 0.9 * limit:
            logger.warning(
                "%s/%s used %d of %d context tokens; lower ollama.max_body_chars or raise num_ctx",
                mail.folder,
                mail.uid,
                used,
                limit,
            )

    def _chat(self, messages: list[dict[str, str]]) -> ollama.ChatResponse:
        try:
            return self._request(messages, think=self._think)
        except ollama.ResponseError as error:
            if self._think is None or "think" not in str(error).lower():
                raise
            logger.debug("%s rejects the think flag, retrying without it", self.model)
            self._think = None
            return self._request(messages, think=None)

    def _request(
        self, messages: list[dict[str, str]], *, think: bool | None
    ) -> ollama.ChatResponse:
        options: dict[str, float | int] = {"temperature": self.config.temperature}
        if self.config.num_ctx:
            # Zero means: keep whatever context the model's own Modelfile declares.
            options["num_ctx"] = self.config.num_ctx
        return self._client.chat(
            model=self.model,
            messages=messages,
            format=_Response.model_json_schema(),
            options=options,
            think=think,
            keep_alive=self.config.keep_alive,
        )

    def unload(self) -> None:
        """Release the model's memory instead of waiting for the keep-alive to expire."""
        try:
            self._client.chat(model=self.model, messages=[], keep_alive=0)
        except (ollama.ResponseError, ConnectionError, TimeoutError) as error:
            logger.debug("Could not unload %s: %s", self.model, error)

    def classify_many(
        self, mails: Iterable[Mail]
    ) -> Iterator[tuple[Mail, Verdict | ClassificationError]]:
        """Classify in parallel, yielding results in input order."""
        mails = list(mails)
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = [pool.submit(self._classify_safely, mail) for mail in mails]
            for mail, future in zip(mails, futures, strict=True):
                yield mail, future.result()

    def _classify_safely(self, mail: Mail) -> Verdict | ClassificationError:
        try:
            return self.classify(mail)
        except ClassificationError as error:
            return error
