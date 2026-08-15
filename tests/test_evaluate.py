import json
import random

import pytest
from imap_tools import FolderInfo, MailboxFolderSelectError

from llm_spam_organizer.evaluate import _sample_uids, build_curves, format_table, load_dataset
from llm_spam_organizer.mail import Mail


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


def test_dataset_round_trip(tmp_path):
    mail = Mail(
        uid="1", folder="Junk", sender="a@b", recipients="c@d", subject="s", date="", body="b"
    )
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({"is_spam": True, **mail.to_dict()}) + "\n")
    (sample,) = load_dataset(path)
    assert sample.is_spam
    assert sample.mail == mail
