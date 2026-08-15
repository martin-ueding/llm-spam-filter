# LLM Spam Organizer

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
$ llm-spam-organizer init-config
```

writes a commented example to `~/.config/llm-spam-organizer/config.toml` with mode
`0600`. The password can be given literally as `password` or, better, as a
`password_command` whose first output line is used, so that it can come from `pass` or
`secret-tool`.

## Moving spam

```console
$ llm-spam-organizer run --dry-run     # classify and print, move nothing
$ llm-spam-organizer run               # move everything at or above the threshold
$ llm-spam-organizer run --watch       # keep running, wait for new mail via IMAP IDLE
```

Only messages with a UID above the highest one already seen are classified. That
watermark lives in `~/.local/state/llm-spam-organizer/state.json`, keyed by account,
folder, and `UIDVALIDITY`, and it only advances on a real run, not on a dry run.
`--reprocess` starts over, `--limit N` restricts a run to the newest N messages.

For a periodic run without `--watch`, a systemd user timer calling
`llm-spam-organizer run` every few minutes works well.

Single files can be classified without touching the server, which is the fastest way to
try a prompt:

```console
$ llm-spam-organizer classify --prompt rubric --show-input message.eml
```

## Comparing models and prompts

```console
$ llm-spam-organizer evaluate
```

fetches the spam folder and a random sample of `evaluation.archive_folders` (spam is the
positive class, archived mail the negative one), caches the extracted messages in
`evaluation/dataset.jsonl`, and runs every combination of `evaluation.models` and
`evaluation.prompts` over them. Results land in `evaluation/scores.json` and
`evaluation/roc.pdf`, plus a summary on standard output:

```
model / prompt                                  AUC   thr*   TPR*   FPR*  FPR@95  fail
--------------------------------------------------------------------------------------
qwen3.5:2b-q4_K_M / detailed                  0.981   0.85  0.964  0.031   0.049     0
qwen3.5:0.8b / concise                        0.943   0.90  0.912  0.078   0.145     2
```

`thr*` maximizes Youden's J and is a reasonable value for `ollama.threshold`, though for
a spam filter the false positive rate matters more than the balanced optimum: `FPR@95`
says how much ham gets misfiled when 95 % of the spam is caught.

The dataset cache and the scores are both reused, and scores are checkpointed after every
combination, so an interrupted run resumes where it stopped. `--refresh` fetches the
dataset again, `--rescore` throws the verdicts away.

Since the sample is drawn from folders the filter already sorted, the labels are only as
good as the sorting that produced them; mail that the old filter got wrong shows up as a
classification error here.

## Notes on running this on a laptop

- Thinking is turned off (`ollama.think = false`). On CPU a reasoning trace costs minutes
  per message and did not produce better verdicts than the direct answer.
- `ollama.concurrency` defaults to 2. More parallel requests grow the KV cache and, on a
  16 GB machine, push the model out of memory rather than making anything faster.
- The evaluation unloads each model before it loads the next one, so only one set of
  weights is resident at a time.
- Only the headers, a short attachment list, and at most `ollama.max_body_chars`
  characters of the body are sent to the model. Attachment payloads never are.

## Development

```console
$ uv run pytest
```
