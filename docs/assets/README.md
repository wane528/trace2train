# Assets status

Date: 2026-08-01

No faithful automated terminal screenshot was produced in this environment, so
`docs/assets/inspect-failure-types.png` has **not** been created.

Reason:

- the reproducible artifact available here is the real markdown export from
  `inspect --demo` at [`examples/sample_report.md`](../../examples/sample_report.md)
- a trustworthy PNG capture still needs a manual terminal screenshot pass

## Pending manual capture

Run this exact command in a UTF-8-capable terminal:

```bash
python -X utf8 -m trace2train.cli inspect --demo
```

Capture target:

- terminal size: `100x30`
- date to record in asset metadata: `2026-08-01`
- crop policy: **crop only** (no retouching, no synthetic overlay, no value edits)

If a PNG is added later, record at minimum:

- app: `trace2train`
- version: `0.1.0`
- capture dimensions
- capture date
- the note that the image is a crop-only screenshot of real CLI output
