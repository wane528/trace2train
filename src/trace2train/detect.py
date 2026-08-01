"""Failure detection.

Two layers, designed to work without an LLM first and optionally with one:

1. Deterministic rules: `error` field present, empty/absent outputs,
   obviously-bad tool calls (empty args), etc.
2. Optional LLM classification (in `attribute.py`) refines *why* it failed.

The output of this stage is a list of failed traces that are worth turning
into training data.
"""

from __future__ import annotations

from collections.abc import Iterable

from .models import Trace

# Patterns that suggest the "failure" is environmental, not behavioral.
# We still keep these traces but tag them so the user can filter later.
ENV_ERROR_HINTS = (
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "network",
    "connection",
    "5xx",
    "500",
    "502",
    "503",
    "api key",
    "unauthorized",
    "authentication",
    "quota",
)

# Tool-call / behavioral failure sub-types (ToolScan-aligned, rules-based).
# Ordered by specificity: the first matching category wins.
FAILURE_TYPE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("wrong_tool", ("wrong tool", "instead of", "incorrect function", "wrong function")),
    ("bad_args", (
        "bad arg", "argument", "missing required", "must be", "invalid parameter",
        "iso 8601", "expected string", "expected int", "type mismatch", "invalid enum",
    )),
    ("format_error", (
        "format error", "invalid format", "expected json", "must be json",
        "not valid json", "malformed", "prose instead", "structured output",
    )),
    ("over_refusal", (
        "over-refusal", "over refusal", "refused", "refusal", "cannot help", "can't help",
    )),
    ("policy_violation", (
        "policy", "system rule", "system instruction", "violat", "without confirm",
        "not allowed",
    )),
    ("lost_context", (
        "lost context", "forgot", "context lost", "dropped", "doesn't remember",
    )),
    ("hallucination", ("hallucinat", "fabricat", "made up", "unsupported claim")),
]


def classify_failure(reasons: list[str], env_only: bool) -> str:
    """Map a failed trace to a ToolScan-style sub-type from its reason text.

    Rules-only (no LLM). Returns 'env_error' for environmental failures and
    'other' when no specific category matches.
    """
    if env_only:
        return "env_error"
    blob = " ".join(reasons).lower()
    for label, hints in FAILURE_TYPE_HINTS:
        if any(h in blob for h in hints):
            return label
    return "other"


class DetectionResult:
    """Outcome of failure detection for a single trace."""

    def __init__(
        self,
        trace: Trace,
        failed: bool,
        reasons: list[str],
        env_only: bool,
        failure_type: str = "other",
    ):
        self.trace = trace
        self.failed = failed
        self.reasons = reasons
        self.env_only = env_only
        self.failure_type = failure_type

    @property
    def trainable(self) -> bool:
        """Failed for a behavioral reason (not purely environmental)."""
        return self.failed and not self.env_only


def _reason_for_error(error: str | None) -> str | None:
    if not error:
        return None
    err = error.lower()
    if any(hint in err for hint in ENV_ERROR_HINTS):
        return None  # environmental; not a behavioral failure
    return error[:500]


def _empty_output_run(run) -> bool:
    """A run that produced no usable output (common silent failure).

    Empty means: none of the known output keys hold a non-empty value, AND
    there is no other truthy content in outputs. A run with {"output": "hi"}
    is NOT empty even though "completion"/"text" are absent.
    """
    outputs = run.outputs or {}
    if not outputs:
        return True

    known_keys = ("output", "completion", "text")
    for key in known_keys:
        val = outputs.get(key)
        if isinstance(val, str) and val.strip():
            return False  # found a usable string output
        if val not in (None, "", [], {}) and not isinstance(val, str):
            return False  # non-string truthy output (e.g. structured)

    # No usable value under known keys; check for any other truthy content.
    for key, val in outputs.items():
        if key in known_keys:
            continue
        if isinstance(val, str) and val.strip():
            return False
        if val not in (None, "", [], {}) and not isinstance(val, str):
            return False

    return True


def detect_failures(traces: Iterable[Trace]) -> list[DetectionResult]:
    """Classify each trace as failed/ok and collect human-readable reasons."""
    results: list[DetectionResult] = []
    for trace in traces:
        reasons: list[str] = []
        env_only = True

        for run in trace.runs:
            if not run.succeeded:
                reason = _reason_for_error(run.error)
                if reason:
                    reasons.append(f"{run.name}: {reason}")
                    env_only = False
                else:
                    # error existed but looks environmental
                    err_hint = run.error[:200] if run.error else "no error message"
                    reasons.append(f"{run.name}: env-ish error: {err_hint}")
            elif _empty_output_run(run) and run.run_type.value in ("llm", "chain"):
                err_hint = run.error[:200] if run.error else ""
                reasons.append(f"{run.name}: empty output (silent failure) {err_hint}".strip())
                env_only = False

        failed = bool(reasons)
        failure_type = classify_failure(reasons, env_only) if failed else "other"
        results.append(DetectionResult(trace, failed, reasons, env_only, failure_type))
    return results
