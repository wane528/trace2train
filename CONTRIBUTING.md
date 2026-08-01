# Contributing to trace2train

Thanks for your interest in improving trace2train. This is a small, honest
open-source project — contributions that keep it truthful, well-tested, and
focused on tool-call / agent-behavior failures are very welcome.

## Development setup

Python 3.11+ is required.

```bash
git clone https://github.com/wane528/trace2train
cd trace2train
pip install -e ".[dev]"
```

## Everyday commands

```bash
# run the full test suite
python -m pytest -q

# lint (and see any style issues)
python -m ruff check .

# build sdist + wheel
python -m build

# try the CLI end to end (no data / no key needed)
trace2train inspect --demo
trace2train convert --demo --no-llm -o out
trace2train review -o out
```

On a non-UTF-8 Windows console, output automatically falls back to ASCII
borders/symbols. For the Unicode box-drawing tables, run under UTF-8:
`python -X utf8 -m trace2train.cli inspect --demo`.

## Pull request checklist

Before opening a PR, please make sure:

- `python -m pytest -q` passes.
- `python -m ruff check .` is clean.
- New behavior is covered by a test (this project follows test-first; write a
  failing test, then the minimal code to pass it).
- No secrets, real API keys, or raw user/customer traces are committed. `.env`,
  pulled snapshots, and generated outputs are git-ignored — keep it that way.
- Documentation claims stay accurate. If you change behavior, update the README
  and any affected doc; do not overstate validation or support status.

## Scope

trace2train deliberately only corrects **behavioral** failures whose right answer
is derivable from the trace (wrong tool, bad args, lost context, wrong format,
over-refusal, plainly wrong common-sense answers). It skips failures that need
external verification instead of fabricating an answer. Please keep new features
inside that scope, or open an issue to discuss first.

## Reporting issues

Open an issue at https://github.com/wane528/trace2train/issues. When reporting a
data or CLI problem, include the command you ran, the exit code, and a **sanitized**
snippet (no credentials, no private trace content).
