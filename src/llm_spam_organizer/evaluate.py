"""Compare models and prompts on the user's own spam and archive folders."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from imap_tools import MailBox, MailboxFolderSelectError

from .classify import ClassificationError, SpamClassifier
from .config import Config
from .mail import Mail, fetch_mails, open_mailbox

logger = logging.getLogger(__name__)

DATASET_NAME = "dataset.jsonl"
SCORES_NAME = "scores.json"
PLOT_NAME = "roc.pdf"


@dataclass(frozen=True, slots=True)
class Sample:
    mail: Mail
    is_spam: bool


@dataclass(frozen=True, slots=True)
class Curve:
    """A ROC curve for one model and prompt combination."""

    model: str
    prompt: str
    fpr: list[float]
    tpr: list[float]
    thresholds: list[float]
    auc: float
    failures: int

    @property
    def label(self) -> str:
        return f"{self.model} / {self.prompt}"

    def best_threshold(self) -> tuple[float, float, float]:
        """Threshold maximizing Youden's J, with its true and false positive rate."""
        index = max(
            range(len(self.thresholds)),
            key=lambda i: self.tpr[i] - self.fpr[i],
        )
        return self.thresholds[index], self.tpr[index], self.fpr[index]

    def fpr_at_tpr(self, target: float) -> float:
        candidates = [f for f, t in zip(self.fpr, self.tpr, strict=True) if t >= target]
        return min(candidates, default=1.0)


def collect_dataset(config: Config, path: Path) -> list[Sample]:
    """Fetch spam and archived mail from the server and cache it as JSON lines."""
    evaluation = config.evaluation
    if not evaluation.archive_folders:
        raise ValueError("Set evaluation.archive_folders to at least one folder")
    rng = random.Random(evaluation.seed)
    samples: list[Sample] = []

    with open_mailbox(config.imap) as mailbox:
        # Sample UIDs before fetching, so that a large archive is not downloaded whole.
        plan = [
            (_sample_uids(mailbox, [config.imap.spam_folder], evaluation.spam_sample, rng), True),
            (_sample_uids(mailbox, evaluation.archive_folders, evaluation.ham_sample, rng), False),
        ]
        for selection, is_spam in plan:
            for folder, uids in selection.items():
                mailbox.folder.set(folder)
                logger.info("Fetching %d message(s) from %s", len(uids), folder)
                samples += [Sample(mail, is_spam) for mail in fetch_mails(mailbox, folder, uids)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps({"is_spam": sample.is_spam, **sample.mail.to_dict()}) + "\n")
    logger.info(
        "Cached %d spam and %d ham message(s) in %s",
        sum(s.is_spam for s in samples),
        sum(not s.is_spam for s in samples),
        path,
    )
    return samples


def _sample_uids(
    mailbox: MailBox, folders: list[str], count: int, rng: random.Random
) -> dict[str, list[str]]:
    """Draw at most `count` UIDs spread over the given folders, grouped by folder."""
    pairs: list[tuple[str, str]] = []
    for folder in folders:
        try:
            mailbox.folder.set(folder)
        except MailboxFolderSelectError:
            available = ", ".join(info.name for info in mailbox.folder.list())
            raise ValueError(
                f"No folder {folder!r} on the server. Available: {available}"
            ) from None
        pairs += [(folder, uid) for uid in mailbox.uids()]
    if len(pairs) > count:
        pairs = rng.sample(pairs, count)
    selection: dict[str, list[str]] = {}
    for folder, uid in sorted(pairs, key=lambda pair: (pair[0], int(pair[1]))):
        selection.setdefault(folder, []).append(uid)
    return selection


def load_dataset(path: Path) -> list[Sample]:
    samples = []
    with path.open() as handle:
        for line in handle:
            data = json.loads(line)
            is_spam = data.pop("is_spam")
            samples.append(Sample(Mail.from_dict(data), is_spam))
    return samples


def score_dataset(
    config: Config,
    samples: list[Sample],
    previous: dict[str, list[dict]] | None = None,
    on_scored: Callable[[dict[str, list[dict]]], None] | None = None,
) -> dict[str, list[dict]]:
    """Run every model and prompt combination over the dataset.

    `on_scored` is called after each combination so that a long run can be interrupted
    and resumed without losing what has already been classified.
    """
    evaluation = config.evaluation
    models = evaluation.models or [config.ollama.model]
    prompts = evaluation.prompts or [config.ollama.prompt]
    scores = dict(previous or {})

    for model in models:
        classifier = None
        for prompt in prompts:
            key = f"{model}|{prompt}"
            if key in scores:
                logger.info("Skipping %s, already scored", key)
                continue
            classifier = SpamClassifier(config.ollama, model=model, prompt=prompt)
            start = time.monotonic()
            scores[key] = _score_one(classifier, samples, key)
            logger.info("%s: %d message(s) in %.0f s", key, len(samples), time.monotonic() - start)
            if on_scored is not None:
                on_scored(scores)
        if classifier is not None:
            # Free the weights before the next model is pulled into memory.
            classifier.unload()
    return scores


def _score_one(classifier: SpamClassifier, samples: list[Sample], key: str) -> list[dict]:
    rows: list[dict] = []
    results = classifier.classify_many(sample.mail for sample in samples)
    for index, (sample, (mail, result)) in enumerate(zip(samples, results, strict=True), start=1):
        row = {"uid": mail.uid, "folder": mail.folder, "is_spam": sample.is_spam}
        if isinstance(result, ClassificationError):
            logger.warning("%s failed on %s/%s: %s", key, mail.folder, mail.uid, result)
        else:
            row |= {
                "subject": mail.subject,
                "sender": mail.sender,
                "spam_probability": result.spam_probability,
                "reason": result.reason,
            }
        rows.append(row)
        if index % 25 == 0:
            logger.info("%s: %d/%d", key, index, len(samples))
    return rows


def build_curves(scores: dict[str, list[dict]]) -> list[Curve]:
    from sklearn.metrics import roc_auc_score, roc_curve

    curves = []
    for key, rows in scores.items():
        model, prompt = key.split("|", 1)
        usable = [row for row in rows if "spam_probability" in row]
        labels = [int(row["is_spam"]) for row in usable]
        probabilities = [row["spam_probability"] for row in usable]
        if len(set(labels)) < 2:
            logger.warning("%s has only one class, skipping ROC", key)
            continue
        fpr, tpr, thresholds = roc_curve(labels, probabilities)
        curves.append(
            Curve(
                model=model,
                prompt=prompt,
                fpr=fpr.tolist(),
                tpr=tpr.tolist(),
                # roc_curve prepends an infinite threshold; clip it for readability.
                thresholds=[min(float(t), 1.0) for t in thresholds],
                auc=float(roc_auc_score(labels, probabilities)),
                failures=len(rows) - len(usable),
            )
        )
    return sorted(curves, key=lambda curve: curve.auc, reverse=True)


def plot_curves(curves: list[Curve], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = ["-", "--", "-.", (0, (3, 1, 1, 1, 1, 1))]
    prompts = sorted({curve.prompt for curve in curves})
    models = sorted({curve.model for curve in curves})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    figure, axes = plt.subplots(figsize=(7.5, 6), layout="constrained")
    for curve in curves:
        # Identical curves would hide each other, so vary style and width as well as color.
        rank = models.index(curve.model)
        axes.plot(
            curve.fpr,
            curve.tpr,
            label=f"{curve.label} (AUC {curve.auc:.3f})",
            color=colors[rank % len(colors)],
            linestyle=styles[prompts.index(curve.prompt) % len(styles)],
            linewidth=2.4 - 0.35 * rank,
            alpha=0.85,
        )
        _, tpr, fpr = curve.best_threshold()
        axes.plot(fpr, tpr, marker="o", markersize=4, color=colors[rank % len(colors)])
    axes.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="chance")
    axes.set_xlabel("False positive rate (ham moved to spam)")
    axes.set_ylabel("True positive rate (spam caught)")
    axes.set_title("Spam classification ROC, dots mark the best threshold")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right", fontsize="small")
    figure.savefig(path)
    plt.close(figure)
    logger.info("Wrote %s", path)


def format_table(curves: list[Curve]) -> str:
    width = max((len(curve.label) for curve in curves), default=14)
    header = (
        f"{'model / prompt':<{width}} {'AUC':>6} {'thr*':>6} "
        f"{'TPR*':>6} {'FPR*':>6} {'FPR@95':>7} {'fail':>5}"
    )
    lines = [header, "-" * len(header)]
    for curve in curves:
        threshold, tpr, fpr = curve.best_threshold()
        lines.append(
            f"{curve.label:<{width}} {curve.auc:>6.3f} {threshold:>6.2f} {tpr:>6.3f} "
            f"{fpr:>6.3f} {curve.fpr_at_tpr(0.95):>7.3f} {curve.failures:>5d}"
        )
    lines.append("")
    lines.append("thr* maximizes Youden's J; FPR@95 is the false positive rate at 95 % recall.")
    return "\n".join(lines)


def evaluate(config: Config, *, refresh: bool = False, rescore: bool = False) -> list[Curve]:
    output = config.evaluation.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = output / DATASET_NAME
    scores_path = output / SCORES_NAME

    if refresh or not dataset_path.exists():
        samples = collect_dataset(config, dataset_path)
    else:
        samples = load_dataset(dataset_path)
        logger.info("Loaded %d cached message(s) from %s", len(samples), dataset_path)

    previous = None
    if scores_path.exists() and not rescore and not refresh:
        previous = json.loads(scores_path.read_text())

    def checkpoint(scores: dict[str, list[dict]]) -> None:
        scores_path.write_text(json.dumps(scores, indent=2))

    scores = score_dataset(config, samples, previous, on_scored=checkpoint)
    checkpoint(scores)

    curves = build_curves(scores)
    if curves:
        plot_curves(curves, output / PLOT_NAME)
    return curves
