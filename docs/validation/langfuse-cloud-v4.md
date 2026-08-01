# Langfuse Cloud v4 validation

Date: 2026-08-01
Region: EU Cloud (`https://cloud.langfuse.com`)
Instance version (from `/api/public/health`): **4.2.0**

Status: **PASSED** — a real, non-empty v2 observations `pull → inspect → convert`
completed against a live Langfuse v4 project with real credentials.

This is a **sanitized aggregate record**. It contains no trace content,
observation IDs, prompts, tool arguments, credentials, or project identifiers —
only counts and structural shapes.

## Method

1. Authenticated against the real Public API (`/api/public/projects` → 200).
2. Seeded synthetic, non-sensitive traces with
   `scripts/seed_langfuse_validation.py`: a wrong-tool behavioral failure, a
   successful generation, and a bad-args multi-observation trace, including
   `ERROR`-level tool observations and structured input/output.
3. Waited for v2 indexing. Note: the v1 observations API showed the seeded data
   almost immediately, while the **v2** observations read path caught up only
   after a delay of several minutes — so the earlier empty v2 result was an
   indexing lag, not a defect, and the pull client handled the empty response
   correctly throughout.
4. Ran the real `trace2train langfuse pull`, then `inspect` and
   `convert --no-llm` on the resulting local snapshot.

## Aggregate results

Pull (`trace2train langfuse pull ... --page-size 3`, cursor-paginated):

- observations pulled: **10**
- snapshot rows written: **10**
- exit code 0, atomic write, no credential or content leakage

Structural audit of the snapshot:

- every row satisfied the required contract (`id`, `traceId`, `startTime`, plus a
  discriminator): **structurally_valid = true**
- observation types present: `AGENT, CHAIN, GENERATION, RETRIEVER, SPAN, TOOL`
- distinct traces: **3**
- `ERROR`-level rows: **2**
- at least one `isRootObservation` root: **yes**
- `GENERATION.input` shape: **str** — confirming the v2 API returns input/output
  as raw strings, which the importer parses (JSON-string → object) as designed.

Inspect (auto-detected format `langfuse`):

- traces: **3**
- failed: **2**
- trainable: **2**
- failure types: `wrong_tool: 1`, `bad_args: 1`

Convert (`--no-llm`):

- SFT records: **2**
- DPO records: **0**
- meta counts: `total_traces=3, failed_traces=2, trainable_traces=2, sft_records=2`
- provenance on generated records:
  - `source_file` matches the local snapshot path: **yes**
  - carries the Langfuse `trace_id` and observation `run_id`: **yes**
  - carries the original error: **yes**
- every generated SFT record had non-empty conversation turns, confirming real v2
  raw-string input/output was correctly surfaced through `Run.messages`.
  **No model extraction change was required.**

## Conclusion

The Langfuse integration works end to end against a real Langfuse Cloud v4
instance and its official Public API v2 observations endpoint: authentication,
cursor pagination, atomic snapshot, auto-detection, trace grouping, failure
detection, and SFT/DPO generation with correct provenance. It is promoted from
Experimental to **Supported** for Langfuse Cloud/self-hosted v4 via the Public
API v2 observations endpoint.

Scope unchanged: no v3 legacy APIs, blob-storage exports, UI-download JSON
shapes, OpenTelemetry ingestion/export, sync/daemon behavior, or write-back.

## Privacy

- No credentials were printed or committed.
- No trace content, prompts, tool arguments, observation IDs, or project IDs are
  recorded.
- The pulled snapshot and generated outputs lived outside the repository, are
  git-ignored, and were deleted after this record was written.
