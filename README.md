# LLM Spam Filter

My mail provider's spam filter sucks and I rely on the spam filter in Thunderbird. That isn't ideal, but better than nothing. Thunderbird's filter is okay, but a small LLM should be even better.

This is a Python project that connects to a server via IMAP and moves spam into the spam folder. It runs the e-mail content (without attachments) through a local LLM (like a Qwen one) via Ollama.

It also has a testing script where it looks at all the e-mails from the spam folder and a sample from the archive folders to generate a ROC curve of various LLMs with different prompts.

Config works via a TOML file where one gives the IMAP credentials.

## Installation

```console
$ uv sync --extra eval
```

The `eval` extra pulls in scikit-learn and matplotlib, which only the `evaluate`
subcommand needs. Ollama has to be running and the models have to be pulled:

```console
$ ollama pull qwen3.5:2b-q4_K_M
```

## Configuration

```console
$ llm-spam-filter init-config
```

writes a commented example to `~/.config/llm-spam-filter/config.toml` with mode
`0600`. The password can be given literally as `password` or, better, as a
`password_command` whose first output line is used, so that it can come from `pass` or
`secret-tool`.

## Moving spam

```console
$ llm-spam-filter run --dry-run     # classify and print, move nothing
$ llm-spam-filter run               # move everything at or above the threshold
$ llm-spam-filter run --watch       # keep running, wait for new mail via IMAP IDLE
```

Only messages with a UID above the highest one already seen are classified. That
watermark lives in `~/.local/state/llm-spam-filter/state.json`, keyed by account,
folder, and `UIDVALIDITY`, and it only advances on a real run, not on a dry run.
`--reprocess` starts over, `--limit N` restricts a run to the newest N messages.

For a periodic run without `--watch`, a systemd user timer calling
`llm-spam-filter run` every few minutes works well.

Single files can be classified without touching the server, which is the fastest way to
try a prompt:

```console
$ llm-spam-filter classify --prompt rubric --show-input message.eml
```

## Comparing models and prompts

```console
$ llm-spam-filter evaluate
```

fetches the spam folder and a random sample of `evaluation.archive_folders` (spam is the
positive class, archived mail the negative one), caches the extracted messages in
`evaluation/dataset.jsonl`, and runs every combination of `evaluation.models` and
`evaluation.prompts` over them. Each verdict is written to its own file under
`evaluation/<model>/<prompt>/<hash>.json`, named after a content hash of the message
(sha256, 12 hex characters), plus `evaluation/roc.pdf` and a summary on standard output:

```
model / prompt                                  AUC   thr*   TPR*   FPR*  FPR@95  fail
--------------------------------------------------------------------------------------
qwen3.5:2b-q4_K_M / detailed                  0.981   0.85  0.964  0.031   0.049     0
qwen3.5:0.8b / concise                        0.943   0.90  0.912  0.078   0.145     2
```

`thr*` maximizes Youden's J and is a reasonable value for `ollama.threshold`, though for
a spam filter the false positive rate matters more than the balanced optimum: `FPR@95`
says how much ham gets misfiled when 95 % of the spam is caught.

The dataset cache and the per-message verdict files are both reused, so a run interrupted
by a crash or an out-of-memory kill resumes by skipping whatever files are already on
disk, down to individual messages rather than whole model/prompt combinations. `--refresh`
fetches the dataset again (the content hash still matches previously scored messages);
`--rescore` throws the verdicts away and reclassifies everything.

Since the sample is drawn from folders the filter already sorted, the labels are only as
good as the sorting that produced them; mail that the old filter got wrong shows up as a
classification error here.

## Notes on running this on a laptop

Thinking is turned off (`ollama.think = false`). On CPU a reasoning trace costs minutes
per message and did not produce a better verdict than the direct answer: `qwen3.5:2b-q4_K_M`
took 5 min 24 s on one message with thinking and 10.6 s without, for the same conclusion.

Only one model is resident at a time. After the last prompt of a model, the evaluation
sends `keep_alive = 0`, which makes Ollama drop the weights immediately instead of holding
them for the keep-alive window. Without that, `OLLAMA_MAX_LOADED_MODELS` (3 by default)
would let several models sit in memory at once. Watching `ollama ps` through a two model
sweep shows `qwen3.5:0.8b` (1.1 GB), then nothing, then `qwen3.5:2b-q4_K_M` (1.6 GB).

`ollama.num_ctx` has to fit the whole prompt, and the interesting number is not how many
characters a mail has but how badly they tokenize. Measured with `qwen3.5:0.8b`:

| content | chars per token |
| --- | --- |
| English prose | 5.3 |
| German prose | 4.8 |
| base64-like payload | 1.9 |

Spam is full of the last kind, so the configuration sizes the context for 1.9. The default
`num_ctx = 8192` covers the default `max_body_chars = 4000` about twice over. Raising
`max_body_chars` without raising `num_ctx` is refused at startup rather than silently
truncating: Ollama drops tokens from the *front* of an overlong prompt, which is exactly
where the system prompt sits. At runtime a message that uses more than 90 % of the context
is logged as a warning.

Context is cheap on these model sizes, so there is no reason to be stingy. For
`qwen3.5:2b-q4_K_M` the resident size grows from 1.6 GB at `num_ctx = 4096` to 1.7 GB at
8192, 1.9 GB at 16384, and 2.1 GB at 32768. Setting `num_ctx = 0` sends no context option
at all and uses whatever the model's own Modelfile declares, which is the point of the
`-32k` tags.

`ollama.concurrency` defaults to 2, which is where the speedup stops. Three messages
through `qwen3.5:0.8b` took 10.8 s at concurrency 1, 7.9 s at 2, and 8.2 s at 3; the CPU
is saturated after that. The resident size did not change under four concurrent requests,
so the setting costs throughput, not memory. Larger models are more compute bound and gain
less: `qwen3.5:2b-q4_K_M` stays near 10 s per message either way.

Only the headers, a short attachment list, and at most `ollama.max_body_chars` characters
of the body are sent to the model. Attachment payloads never are.

## Development

```console
$ uv run pytest
$ uv run ruff check .
```

Commits follow [Conventional Commits](https://www.conventionalcommits.org/). On every push to
`main`, CI runs lint and tests; if they pass, [Commitizen](https://commitizen-tools.github.io/commitizen/)
checks whether the commits since the last tag warrant a release, and if so bumps the version in
`pyproject.toml`, updates `CHANGELOG.md`, tags the commit, and creates a GitHub release. Publishing
that release to PyPI happens in a separate workflow, via PyPI's trusted publishing (OIDC), so no
API token is stored in this repository.
