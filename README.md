# mdidx

`mdidx` is a Rust CLI that indexes Markdown files into LanceDB so you can search and synthesize across one or more Obsidian vaults with a consistent, repeatable workflow.

Highlights:
- Multi‑vault indexing in one command (or continuous watch mode).
- Semantic vector search, BM25 full‑text search, and hybrid RRF search.
- MCP servers (stdio + Streamable HTTP) with tools for search, stats, and note synthesis.
- Optional synthesis pipeline: classify + upsert into a distilled vault with strict frontmatter and idempotency.
- Incremental indexing with mtime/checksum change detection and optional FTS refresh.

Current build target: macOS only.

## Usage

Most defaults are sane — you usually don’t need a long command. See the **Tutorial** section below for the recommended, minimal workflows.

Example (full options shown for reference):

```sh
mdidx index <path> [<path> ...] \
  --db lancedb \
  --provider local \
  --model nomic \
  --algorithm mtime \
  --chunk-size 1000 \
  --chunk-overlap 100
```

By default, indexing prints per-file progress. Use `--quiet` to suppress it.

Default embeddings:
- Provider: `local`
- Model: `nomic` (`nomic-embed-text-v1.5`, 768 dims)
- If you choose OpenAI (`--provider openai`), the default model is `text-embedding-3-small`.

For better performance and quality, we recommend using OpenAI embeddings. When you do, the default model is `text-embedding-3-small` unless you override it.

If you switch embedding dimensions/models, either point at a new `--db` directory or pass `--reset --confirm` to drop and recreate the `chunks`/`files` tables.

Indexing writes `mdidx-config.json` into the DB directory with the embedding config. Searches (CLI + MCP) read this file and ignore client-side embedding settings when it exists.

Search (JSON output by default):

```sh
mdidx search "vector database" --limit 5
```

Human output:

```sh
mdidx search "vector database" --output human
```

BM25 (lexical) search:

```sh
mdidx search "vector database" --mode bm25
```

Hybrid (vector + BM25) search:

```sh
mdidx search "vector database" --mode hybrid
```

Stats:

```sh
mdidx stats --db lancedb
```

## Full-text index (FTS)

Build a BM25 full-text index on `chunks.content`:

```sh
mdidx fts-build
```

Refresh the index after incremental updates:

```sh
mdidx fts-refresh
```

You can also refresh right after indexing:

```sh
mdidx index ~/Notes ~/Docs --fts-refresh
```

Progress reporting:

```sh
mdidx fts-build --progress-interval-secs 2 --wait-timeout-secs 3600
```

## Watch mode

Watch multiple directories and update the index on change:

```sh
mdidx watch ~/Notes ~/Docs --debounce-ms 750
```

If the database is empty, run `mdidx index` once to seed it before watching.

To keep the FTS index in sync while watching:

```sh
mdidx watch ~/Notes ~/Docs --fts-refresh-interval-secs 60
```

## launchd (macOS)

Template `~/Library/LaunchAgents/com.mdidx.watch.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.mdidx.watch</string>
    <key>ProgramArguments</key>
    <array>
      <string>/usr/local/bin/mdidx</string>
      <string>watch</string>
      <string>/Users/you/Notes</string>
      <string>/Users/you/Docs</string>
      <string>--debounce-ms</string>
      <string>750</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/Users/you/Library/Logs/mdidx-watch.log</string>
    <key>StandardErrorPath</key>
    <string>/Users/you/Library/Logs/mdidx-watch.err.log</string>
    <key>EnvironmentVariables</key>
    <dict>
      <key>OPENAI_API_KEY</key>
      <string>YOUR_KEY</string>
    </dict>
  </dict>
</plist>
```

Load/unload:

```sh
launchctl load -w ~/Library/LaunchAgents/com.mdidx.watch.plist
launchctl unload -w ~/Library/LaunchAgents/com.mdidx.watch.plist
```

## Data location

By default, `mdidx` stores its database and embedding cache under the user data directory:
- macOS: `~/Library/Application Support/mdidx`
- Linux: `~/.local/share/mdidx` (or `$XDG_DATA_HOME/mdidx` if set)
- Windows: `%APPDATA%\\mdidx` (falls back to `%LOCALAPPDATA%` if needed)

Use `--db` to point at a different database directory.

## MCP server

Run the stdio MCP server:

```sh
mdidx-mcp
```

Tools:
- `search_chunks` for vector search.
- `search_chunks_bm25` for BM25 full-text search.
- `search_chunks_hybrid` for hybrid vector + BM25 search (RRF).
- `stats_index` for index stats.
- `get_file_content` to read full file content from disk (supports `max_lines` and `start_line`).
- `synth_note_upsert` to write a synthesized note into a configured vault (strict frontmatter + optional indexing).
- `synth_classify_path` to classify a note into category/domain/subpath and normalized tags (no file writes).

## Vault configuration

Register a synthesis vault path (used by MCP synthesis tools):

```sh
mdidx vault set distilled /path/to/DistilledVault
```

List configured vaults:

```sh
mdidx vault list
```

## AI configuration

Set the model used for AI classification:

```sh
mdidx config set --classify-model gpt-4o-mini
```

Show current AI config:

```sh
mdidx config show
```

Environment variable override:

- `MDIDX_CLASSIFY_MODEL` overrides the config file for classification
- `OPENAI_API_KEY` is required for AI classification

## Optional note-taking skill

This repo includes an optional skill that encodes the glue logic for reliable, repeatable synthesis into the distilled vault.

Skill definition:
- `skills/note-taking/mdidx-synthesis-pipeline/SKILL.md`

What it does:
- Standardizes the synthesis workflow (draft → classify → upsert → report).
- Enforces required frontmatter fields and sections.
- Uses idempotency keys for safe retries.
- Provides a consistent response template for status reporting.

When to use it:
- User asks to synthesize a conversation into a distilled/Obsidian note.
- User asks to classify and persist a note using mdidx tools.

Note: There is also a general Obsidian skill at `skills/note-taking/obsidian/SKILL.md` for manual note operations (read/list/create/append). It is **not required** for the mdidx synthesis pipeline, but can be useful for ad‑hoc edits outside mdidx.

## Skill trigger note

The `mdidx-synthesis-pipeline` skill will only auto‑run if your client has a skill router that maps phrases like “synthesize now” to this skill. If your client doesn’t support routing, you must invoke the skill explicitly or call the MCP tools directly.

## Search modes

**Vector (semantic) search** uses embeddings to find meaning‑similar text. It is good for paraphrases and synonyms, even when the exact words are not present.

**BM25 (lexical) search** uses the FTS index over `chunks.content` to rank by exact term matches. It is good for precise keywords, identifiers, and phrases.

**Hybrid search (vector + BM25)** combines both signals and uses Reciprocal Rank Fusion (RRF) to rerank results.

### Hybrid flow

```text
query text
   |
   |--(A) Vector search --> Top‑K by _distance
   |
   |--(B) BM25 search  --> Top‑K by _score
   |
   +--> RRF merge (combine ranks from A + B)
           |
           v
     Final top‑K by _relevance_score
```

Steps:
1. Embed the query text and run vector search on `chunks.vector` to get candidates with `_distance`.
2. Run BM25 search on `chunks.content` to get candidates with `_score`.
3. Merge the two ranked lists and apply RRF (default `k=60`) to compute `_relevance_score`.
4. Return the top‑K results, along with `_distance`, `_score`, and `_relevance_score`.

### Streamable HTTP

Run the Streamable HTTP server:

```sh
mdidx-mcp-http --host 127.0.0.1 --port 8080 --endpoint /mcp
```

Endpoints:
- `POST /mcp` Streamable HTTP JSON-RPC endpoint (SSE not supported; `GET /mcp` returns 405).
- `POST /rpc` legacy JSON-RPC endpoint for compatibility.
- `GET /tools` list tools.
- `POST /call` direct tool call.
- `GET /health` status.
- `GET /.well-known/mcp.json` discovery metadata.

Origin requests are validated and only `localhost` / `127.0.0.1` are allowed.

Filter examples:

```sh
# Only search within a specific file path
mdidx search "vector database" --filter "file_path = '/path/to/file.md'"

# Apply filter after vector search
mdidx search "vector database" --filter "mtime > 1700000000" --postfilter
```

To perform in-depth testing of the stdio mcp server:

```sh
npx @modelcontextprotocol/inspector ./target/debug/mdidx-mcp
```

Confirm with Claude Desktop:

```sh
vi ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

For instance:

```json
  ...
  "mcpServers": {
    "mdidx": {
      "command": "/Users/chris/Projects/mdidx/target/debug/mdidx-mcp",
      "env": {
        "OPENAI_API_KEY": "sk-xxxxxx"
      }
    }
  ...
```


### Change detection

- `mtime`: compare the stored `mtime` to the current file `mtime` and reindex when it changes.
- `checksum`: read the file and compare a SHA-256 checksum to the stored checksum.

### Tables

- `chunks`: one row per chunk (`file_path`, `chunk_index`, `content`, `checksum`, `mtime`, `vector`).
- `files`: one row per file (`file_path`, `checksum`, `mtime`, `chunk_count`, `indexed_at`).

Files that disappear from the scan path are deleted from both tables.

## Embeddings

`mdidx` uses `fastembed` to run local embedding models and will download weights on first use.

### Models

- `nomic` (default): `nomic-embed-text-v1.5` (768 dims). Requires instruction prefixes.
- `bge-m3`: `BAAI/bge-m3` (1024 dims). No instruction prefix required.

You can override the dimension with `--dim`, but it must match the model output dimension.

Example:

```sh
mdidx index ~/Docs --model bge-m3
```

## OpenAI embeddings

To use OpenAI embeddings instead of local models, set `OPENAI_API_KEY` and pass `--provider openai`.

Supported models:
- `text-embedding-3-small` (1536 dims)
- `text-embedding-3-large` (3072 dims)

Example:

```sh
export OPENAI_API_KEY=...
mdidx index ~/Docs --provider openai --openai-model text-embedding-3-small
```

# Tutorial

We are going to index, update and use both our regular vault, and a "distilled" vault where we store synthesized notes.
We will be sure to watch and query both, thus closing the knowledge loop.

First indexing run:

```sh
mdidx index ~/Documents/Obsidian/CFR ~/Documents/Obsidian/Distilled --provider openai --openai-model text-embedding-3-small
```

Subsequent, regular indexing:

```sh
mdidx index ~/Documents/Obsidian/CFR ~/Documents/Obsidian/Distilled
```

We will not need to keep running the above command if we run `watch` at all time, but it does not hurt to run `index` again if we forgot to leave `watch` running:

```sh
mdidx watch --fts-refresh-interval-secs 60 ~/Documents/Obsidian/CFR ~/Documents/Obsidian/Distilled
```

So, this is totally optional but... if you want to maintain a vault where you can store synthesized notes generated during LLM research, you will need to declare it:

```sh
mdidx vault set distilled /Users/chris/Documents/Obsidian/Distilled
```

From now on, any MCP call to `synth_classify_path` and `synth_note_upsert` will work in this vault.

To include this vault in search, make sure you index it (as shown above) into the same LanceDB. There is no weighting/boosting logic today.

Now, let's copy the skill to the agent's `skills` directory (and enable it if  the agent does not automatically do it)

If your client supports a skill router, map a phrase like `synthesize now` to the `mdidx-synthesis-pipeline` skill. Otherwise you must invoke the skill explicitly or call the MCP tools directly.

With routing enabled, synthesizing a conversation should be as simple as:

> synthesize now

but this varies between clients and agents (See previous trigger explanation)
