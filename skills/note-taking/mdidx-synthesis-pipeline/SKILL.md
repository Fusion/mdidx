---
name: mdidx-synthesis-pipeline
description: Use this skill whenever the user asks to synthesize a conversation into an Obsidian/distilled vault note, classify hierarchy/tags, or persist/update session knowledge via mdidx. Always use for requests like "synthesize now", "save this to vault", "classify and upsert", or "turn this chat into a note".
---

# mdidx Synthesis Pipeline

Create a high-signal synthesized note from the current conversation, classify it into the distilled vault hierarchy, and persist it with idempotent upsert semantics.

## When to use
Use this skill whenever the user wants to:
- Turn chat output into a reusable document
- Save synthesized knowledge into a dedicated Obsidian vault
- Auto-assign hierarchy and tags
- Re-run safe updates without duplicate notes

## Required workflow (always)
1. **Synthesize content first** into a canonical markdown draft.
2. **Persist note** using `mcp_mdidx_synth_note_upsert` with AI pathing.
3. **Optionally classify-only** with `mcp_mdidx_synth_classify_path` when you need a dry-run path/tag suggestion before writing.
4. **Report result** with operation type + final path + indexing status.

Use `synth_note_upsert` as the primary path for end-to-end classify+persist.

## Canonical note structure
Produce markdown body with these sections (as applicable):
- `## TL;DR`
- `## Key Decisions`
- `## Supporting Reasoning`
- `## Open Questions`
- `## Action Items`
- `## References`

## Required frontmatter fields
Ensure these are present in the upsert payload (all are strictly required by the mdidx schema):
- `title`
- `date` (`YYYY-MM-DD`)
- `topic`
- `summary` (concise 1-2 sentence overview)
- `status` (`draft|stable`)
- `tags` (Minimum **3** tags required)
- `source_sessions` (Minimum **1** item required, e.g., `["current_chat"]`)
- `confidence` (`low|med|high`)
- `markdown_body` (The full content of the note)

## Options Configuration
When calling `mcp_mdidx_synth_note_upsert`, always include these options to ensure reliability:
- `mode`: `"create_or_update"` (default)
- `path_strategy`: `"ai_suggested"` (to use mdidx's auto-classification)
- `refresh_index`: `false` (IMPORTANT: set to `false` for large vaults to prevent TimeoutErrors during the write)

## Routing policy
Map note intent to default category:
- How-to/procedure => `20_Playbooks`
- Architecture/tradeoffs/system design => `30_Architecture`
- Concept/domain summaries => `10_Topics`
- Durable factual references => `90_Reference`
- Uncertain/mixed/low confidence => `00_Inbox`

If classifier confidence is low, route to `00_Inbox` and include alternatives in response.

## Idempotency policy
Use stable keys for safe retries:
- Classification key pattern: `classify:<session_or_topic_slug>:v1`
- Upsert key pattern: `upsert:<session_or_topic_slug>:v1`

If user requests a revision to the same synthesis, increment version suffix (`v2`, `v3`) only when intended; otherwise keep key stable for overwrite/merge semantics.

## Execution checklist
Before persisting, verify:
- Title is specific and non-generic
- Tags are normalized (`lowercase`, kebab/snake-safe)
- Source session identifiers are included
- Body is concise and de-duplicated

After persisting, return:
- `operation` (`created|updated|noop`)
- `final_path`
- `note_id` (if available)
- indexing outcome/stat summary
- any warnings

## User-facing response template
Use short confirmation format:

- ✅ Synthesized + classified + persisted
- **Operation:** <created/updated/noop>
- **Path:** `<final_path>`
- **Tags:** <tag list>
- **Index:** <executed/scope/stats>
- **Notes:** <warnings or "none">

## Failure handling
- If classification fails: fallback to `00_Inbox` and persist with conservative tags.
- If upsert fails: return exact error code/message and suggest a retry with same idempotency key.
- If index refresh fails but write succeeded: report partial success clearly.
