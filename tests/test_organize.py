import pytest

from llm_spam_organizer.classify import ClassificationError, Verdict
from llm_spam_organizer.config import parse_config
from llm_spam_organizer.mail import Mail
from llm_spam_organizer.organize import State, _process

RAW = {"imap": {"host": "h", "username": "u", "password": "p"}, "ollama": {"threshold": 0.7}}


class FakeMailbox:
    def __init__(self, mails: list[Mail]) -> None:
        self.mails = mails
        self.moved: list[tuple[list[str], str]] = []

    def uids(self) -> list[str]:
        return [mail.uid for mail in self.mails]

    def fetch(self, criteria, **kwargs):
        raise AssertionError("fetch_mails is patched in these tests")

    def move(self, uids, destination):
        self.moved.append((list(uids), destination))


class FakeClassifier:
    model = "fake"

    def __init__(self, probabilities: dict[str, float | ClassificationError]) -> None:
        self.probabilities = probabilities

    def classify_many(self, mails):
        for mail in mails:
            outcome = self.probabilities[mail.uid]
            if isinstance(outcome, ClassificationError):
                yield mail, outcome
            else:
                yield mail, Verdict(outcome, "because", self.model, "concise")


def mail(uid: str) -> Mail:
    return Mail(
        uid=uid, folder="INBOX", sender="a@b", recipients="c@d", subject=uid, date="", body=""
    )


@pytest.fixture
def setup(monkeypatch, tmp_path):
    def run(uids: list[str], probabilities, *, stored=0, **kwargs):
        mails = [mail(uid) for uid in uids]
        mailbox = FakeMailbox(mails)
        monkeypatch.setattr(
            "llm_spam_organizer.organize.fetch_mails",
            lambda box, folder, wanted: [m for m in mails if m.uid in wanted],
        )
        state = State(tmp_path / "state.json")
        if stored:
            state.set("key", stored)
        run_result = _process(
            mailbox,
            parse_config(RAW),
            FakeClassifier(probabilities),
            state,
            "key",
            dry_run=kwargs.pop("dry_run", False),
            **kwargs,
        )
        return mailbox, state, run_result

    return run


def test_only_messages_above_the_threshold_are_moved(setup):
    mailbox, state, run = setup(["1", "2", "3"], {"1": 0.9, "2": 0.1, "3": 0.71})
    assert mailbox.moved == [(["1", "3"], "Junk")]
    assert (run.seen, run.moved, run.failed) == (3, 2, 0)
    assert state.get("key") == 3


def test_known_uids_are_not_classified_again(setup):
    mailbox, state, run = setup(["1", "2", "3"], {"3": 0.9}, stored=2)
    assert mailbox.moved == [(["3"], "Junk")]
    assert run.seen == 1


def test_dry_run_neither_moves_nor_advances_the_state(setup):
    mailbox, state, run = setup(["1"], {"1": 0.9}, dry_run=True)
    assert mailbox.moved == []
    assert state.get("key") == 0
    assert run.moved == 1


def test_classification_failures_leave_the_message_alone(setup):
    mailbox, state, run = setup(["1", "2"], {"1": ClassificationError("boom"), "2": 0.95})
    assert mailbox.moved == [(["2"], "Junk")]
    assert (run.seen, run.moved, run.failed) == (2, 1, 1)


def test_limit_keeps_the_newest_messages(setup):
    mailbox, state, run = setup(["1", "2", "10"], {"10": 0.1}, limit=1)
    assert run.seen == 1
    assert state.get("key") == 10


def test_reprocess_ignores_the_stored_uid(setup):
    mailbox, state, run = setup(["1", "2"], {"1": 0.1, "2": 0.1}, stored=2, reprocess=True)
    assert run.seen == 2


def test_state_never_moves_backwards(tmp_path):
    state = State(tmp_path / "state.json")
    state.set("key", 10)
    state.set("key", 5)
    assert State(tmp_path / "state.json").get("key") == 10
