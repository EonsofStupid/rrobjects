# RRO as Claude Code's memory

The dogfood: clyffy's own recall spine wired into Claude so it stops forgetting.
If RRO does not earn its place here, it does not earn its place anywhere.

## The loop

1. `rro-recall.py` runs on **`UserPromptSubmit`**. It asks the node what it
   remembers about your prompt and injects the hits as context.
2. The same hook **captures** the prompt back into the estate, so the next
   session recalls what this one was told.

It closes. Measured, on a real estate with Qwen3-Embedding-4B:

```
prompt: "should I just remove this unused helper function to clean things up?"
recall: [prompt_0503aec4] "You aggressively do not deprecate. ... Review dead
         code and classify it before deleting anything ..."   <- rank 1
```

Almost no shared vocabulary between the two. That is the entire point: the
correction surfaces because it *means* the same thing, not because it repeats a
word.

## Install

```sh
cargo build --release --bin rro -p rro-engine --bin rro-mcp -p rro-client
install -Dm644 integrations/claude-code/rro.service ~/.config/systemd/user/rro.service
install -Dm755 integrations/claude-code/rro-recall.py ~/.claude/hooks/rro-recall.py
systemctl --user enable --now rro.service
```

Then register the hook in `~/.claude/settings.json`:

```json
{ "hooks": { "UserPromptSubmit": [ { "hooks": [
  { "type": "command", "command": "/home/YOU/.claude/hooks/rro-recall.py" } ] } ] } }
```

And, for tool access to the same node, `.mcp.json` in your project:

```json
{ "mcpServers": { "rro": {
  "command": "/path/to/rrobjects/target/release/rro-mcp",
  "env": { "RRO_ADDR": "127.0.0.1:7878" } } } }
```

## Two settings that are not defaults, on purpose

**`RRO_RERANKER=identity`.** The default is `LexicalReranker`, and over a hybrid
store it double-counts the lexical signal and re-sorts by the weaker retriever.
Live, it scored the semantically correct document **0.0000** and sank it below an
unrelated document that happened to share a word — see `docs/BENCHMARKS_REAL.md`
Finding 4. A memory hook also cannot afford a ~1 s cross-encoder on every prompt;
identity is 59 ms end to end.

**`RRO_ESTATE` set.** Without it the node is in-memory and seeds a demo corpus.
With it, the estate is yours and is never seeded (`estate_is_yours.rs`).

## Known limits — read these before trusting it

- **There is no relevance gate.** ANN returns *k* nearest neighbours however
  distant, RRF scores are `1/(60+rank)` and carry no magnitude, and the readiness
  verdict is lexical — so an irrelevant prompt still recalls its four nearest
  memories. On a small estate that is harmless noise; it will not stay harmless.
  Fixing it needs score-magnitude fusion (DBSF) or a server-side-embed +
  `score_threshold` path, neither of which exists yet.
- **Capture is gated by a crude heuristic** (machine-text rejection + length +
  a stoplist), not by RRD. RRD's gate ladder is exactly the right tool and is
  already wired into `ask` — but not into `index`. That is the honest fix.
  The first cut of this heuristic checked only length, and machine text walked
  straight in: `<task-notification>` blocks were stored as `operator_prompt` and
  recalled back as "prior facts, decisions and operator corrections". They are
  none of those. If you see the estate quoting the harness at you, that is this
  filter failing — widen `worth_remembering`, do not widen the recall.
- **The readiness label in the block is currently meaningless** (it reads
  `insufficient @ 0.00` even for a perfect hit) for the same lexical reason.
