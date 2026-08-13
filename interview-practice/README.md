# Interview Practice — Coding + Math

This is the training half of
[Interview Bank II · Coding + Math](https://jxzhangjhu.github.io/blog/2026/interview-coding/).
The article explains each implementation; this directory tests whether you can produce
it from a stub under a clock.

The prompts are practice exercises, not official company question banks. A lab name in
`reported` means only that a public anecdotal account associated a similar prompt with
that lab.

## Setup

```bash
git clone https://github.com/jxzhangjhu/jxzhangjhu.github.io.git
cd jxzhangjhu.github.io/interview-practice
python3 -m pip install numpy pytest torch
```

## Daily workflow

[`run.py`](run.py) is the entry point; [`problems.py`](problems.py) is the single source
of truth for titles, budgets, cold-start membership, and cautious attribution labels.

```bash
python run.py                  # list every problem, drill, and local attempt
python run.py list             # explicit list form
python run.py p01              # time p01 and run its focused test file
python run.py mha              # names work too
python run.py p01 --no-timer   # run without the start prompt or attempt log
python run.py --cold           # run the weekly from-empty-file set
python run.py --drill d09      # time the miniGPT debugging drill
python run.py --reset p01      # restore a problem stub
python run.py --reset d09      # restore all planted bugs in a drill
```

For a normal problem:

1. Open its file in [`stubs/`](stubs/).
2. Start with `python run.py pNN`.
3. If stuck, reveal exactly one level in [`hints/`](hints/).
4. Read the failure as a violated property, not as a line-by-line recipe.
5. Reset and repeat later from an empty implementation.

For a debug drill, edit [`drills/`](drills/) instead. The two longer drills are:

- [`d09_minigpt.py`](drills/d09_minigpt.py): an OpenAI-style miniGPT bug hunt plus a
  KV-cache follow-up.
- [`d10_grpo_loop.py`](drills/d10_grpo_loop.py): an Anthropic-style GRPO-loop bug hunt.

Both are syntheses of anecdotal public reports, not verbatim reproductions. The micro
drills each plant one wrong line. Every drill has three hint levels and a reset copy.

## What not to open while practising

[`reference.py`](reference.py) and `drills/.solutions/` contain the answers. The focused
tests may compare numerically against the reference, but their failure messages are
written around observable symptoms. Use the reference only after an attempt, to review
the complete path.

Your timing history is stored in `attempts.local.json`, which is gitignored.

## Maintainer checks

```bash
python reference.py             # independent numerical/property checks
python _validate.py             # every problem test against its reference answer
python _validate.py --drills    # every drill test against its fixed version
python run.py list
```

`_validate.py` also verifies that every metadata row has a stub, pristine reset copy,
test, solution where applicable, and exactly three hint levels.
