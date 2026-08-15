import json
import random

import pytest
from imap_tools import FolderInfo, MailboxFolderSelectError

from llm_spam_filter.classify import ClassificationError, Verdict
from llm_spam_filter.evaluate import (
    Sample,
    _message_hash,
    _sample_uids,
    _score_combo,
    build_curves,
    format_table,
    load_dataset,
)
from llm_spam_filter.mail import Mail


class FakeMailbox:
    def __init__(self, folders: dict[str, list[str]]) -> None:
        self.folders = folders
        self.current = ""
        self.folder = self

    def set(self, name: str) -> None:
        if name not in self.folders:
            raise MailboxFolderSelectError(("NO", [name.encode()]), "OK")
        self.current = name

    def list(self):
        return [FolderInfo(name, "/", ()) for name in self.folders]

    def uids(self) -> list[str]:
        return self.folders[self.current]


def test_sampling_takes_everything_when_the_folders_are_small():
    mailbox = FakeMailbox({"A": ["1", "2"], "B": ["10", "9"]})
    assert _sample_uids(mailbox, ["A", "B"], 100, random.Random(0)) == {
        "A": ["1", "2"],
        "B": ["9", "10"],
    }


def test_sampling_is_capped_and_reproducible():
    mailbox = FakeMailbox({"A": [str(uid) for uid in range(100)]})
    first = _sample_uids(mailbox, ["A"], 10, random.Random(7))
    second = _sample_uids(mailbox, ["A"], 10, random.Random(7))
    assert first == second
    assert len(first["A"]) == 10


def test_missing_folder_is_reported_with_the_available_ones():
    mailbox = FakeMailbox({"A": ["1"]})
    with pytest.raises(ValueError, match="Available: A"):
        _sample_uids(mailbox, ["Nope"], 10, random.Random(0))


def rows(probabilities: list[tuple[float, bool]]) -> list[dict]:
    return [
        {"uid": str(index), "folder": "X", "is_spam": is_spam, "spam_probability": probability}
        for index, (probability, is_spam) in enumerate(probabilities)
    ]


def test_perfect_separation_gives_auc_one():
    scores = {"m|p": rows([(0.1, False), (0.2, False), (0.9, True), (0.95, True)])}
    (curve,) = build_curves(scores)
    assert curve.auc == 1.0
    threshold, tpr, fpr = curve.best_threshold()
    assert (tpr, fpr) == (1.0, 0.0)
    assert 0.2 < threshold <= 0.9


def test_curves_are_sorted_by_auc_and_count_failures():
    scores = {
        "good|p": rows([(0.1, False), (0.9, True)]),
        "bad|p": rows([(0.9, False), (0.1, True)]) + [{"uid": "9", "is_spam": True}],
    }
    curves = build_curves(scores)
    assert [curve.model for curve in curves] == ["good", "bad"]
    assert curves[1].failures == 1
    assert "AUC" in format_table(curves)


def test_single_class_is_skipped():
    assert build_curves({"m|p": rows([(0.1, False), (0.2, False)])}) == []


def make_mail(uid: str, subject: str) -> Mail:
    return Mail(
        uid=uid, folder="Junk", sender="a@b", recipients="c@d", subject=subject, date="", body="b"
    )


class FakeClassifier:
    """Records which mails it was asked to classify and returns a scripted verdict for each."""

    def __init__(self, outcomes: dict[str, Verdict | ClassificationError]) -> None:
        self.outcomes = outcomes
        self.asked: list[str] = []

    def classify_many(self, mails):
        mails = list(mails)
        self.asked += [mail.subject for mail in mails]
        for mail in mails:
            yield mail, self.outcomes[mail.subject]


def test_message_hash_ignores_uid_and_folder_but_not_content():
    a = make_mail("1", "hello")
    b = make_mail("2", "hello")
    c = make_mail("1", "different")
    assert _message_hash(a) == _message_hash(b)
    assert _message_hash(a) != _message_hash(c)


def test_score_combo_writes_one_file_per_message_and_skips_them_next_time(tmp_path):
    samples = [Sample(make_mail("1", "spam mail"), True), Sample(make_mail("2", "ham mail"), False)]
    outcomes = {
        "spam mail": Verdict(0.9, "looks spammy", "m", "p"),
        "ham mail": Verdict(0.1, "looks fine", "m", "p"),
    }
    classifier = FakeClassifier(outcomes)
    rows = _score_combo(classifier, samples, tmp_path, "m|p", rescore=False)

    assert {row["spam_probability"] for row in rows} == {0.9, 0.1}
    written = sorted(p.name for p in tmp_path.glob("*.json"))
    assert written == sorted(f"{_message_hash(s.mail)}.json" for s in samples)
    assert classifier.asked == ["spam mail", "ham mail"]

    # A second run must not ask the classifier again; everything is cached on disk.
    second_classifier = FakeClassifier(outcomes)
    second_rows = _score_combo(second_classifier, samples, tmp_path, "m|p", rescore=False)
    assert second_rows == rows
    assert second_classifier.asked == []


def test_score_combo_does_not_persist_failures_so_they_retry(tmp_path):
    samples = [Sample(make_mail("1", "flaky"), True)]
    failing = FakeClassifier({"flaky": ClassificationError("timeout")})
    rows = _score_combo(failing, samples, tmp_path, "m|p", rescore=False)
    assert "spam_probability" not in rows[0]
    assert list(tmp_path.glob("*.json")) == []

    recovered = FakeClassifier({"flaky": Verdict(0.7, "ok now", "m", "p")})
    rows = _score_combo(recovered, samples, tmp_path, "m|p", rescore=False)
    assert rows[0]["spam_probability"] == 0.7
    assert recovered.asked == ["flaky"]


def test_score_combo_rescore_ignores_cache(tmp_path):
    samples = [Sample(make_mail("1", "mail"), True)]
    first = FakeClassifier({"mail": Verdict(0.2, "first", "m", "p")})
    _score_combo(first, samples, tmp_path, "m|p", rescore=False)

    second = FakeClassifier({"mail": Verdict(0.8, "second", "m", "p")})
    rows = _score_combo(second, samples, tmp_path, "m|p", rescore=True)
    assert second.asked == ["mail"]
    assert rows[0]["spam_probability"] == 0.8


def test_dataset_round_trip(tmp_path):
    mail = Mail(
        uid="1", folder="Junk", sender="a@b", recipients="c@d", subject="s", date="", body="b"
    )
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({"is_spam": True, **mail.to_dict()}) + "\n")
    (sample,) = load_dataset(path)
    assert sample.is_spam
    assert sample.mail == mail
