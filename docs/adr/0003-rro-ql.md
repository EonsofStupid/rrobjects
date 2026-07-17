# ADR-0003 — RRQL: a text query language for RRO

_2026-07-16. Status: **proposed** — the COSTAR block below needs operator
confirmation before `crates/rro-ql/` is created (`clyffy/docs/PLANNING_DISCIPLINE.md`:
"No new crate, top-level folder, or spec doc is created without passing the gate").
Everything else in Phase A was authored UNDER existing boundaries and needed no gate._

## Why this is the biggest parity gap

RRO is meant to be turnkey the way Qdrant, Chroma and SurrealDB are turnkey. All
three ship a way to *send a query as text*:

| engine | text surface |
|---|---|
| SurrealDB | SurrealQL — `SELECT … WHERE`, DEFINE, RELATE, LIVE |
| Qdrant | a JSON filter DSL over HTTP/gRPC |
| Chroma | `where` / `where_document` clauses |
| **RRO today** | **typed Rust structs, and nothing else** |

`EstateQuery`/`Filter` are good and tested — but you can only build them by
linking the crate. There is no way to hand RRO a query as text, which means no
CLI, no REST body, no MCP `rro_sql` tool, and nothing a non-Rust caller can use.
`connectome` (the SurrealDB fork RRO replaces) had all of SurrealQL; this is the
single largest thing RRO must author fresh to stand in for it.

## COSTAR

**Context.** Boundaries ruled out, and why:
- `rro-core` — owns the *typed contract* (`Filter`, `Condition`, `EstateQuery`)
  and is depended on by every other crate. Putting a lexer/parser here forces a
  text front-end onto `embedder`, `recall`, `reranker` — crates that must never
  care that text queries exist. It also inverts the DAG's spirit: the innermost
  crate would grow the outermost concern.
- `connxism` — owns *execution over RocksDB*. A parser here couples the language
  to one storage engine; RRQL must compile to `EstateQuery` and stay executable
  by any `Recall` impl (including `FlatRecall`).
- `rro-engine` — owns *composition + the a2a surface*. A language living here
  could not be used by `rro-client` (which does not depend on the engine) or by a
  future `rro-http`, so the MCP `rro_sql` tool could not reach it.
- A module under an existing crate — the same coupling as above, plus a parser
  is a genuinely separable concern with its own test surface (proptest over
  random ASTs) and its own optional-dependency story.

**Objective.** ONE responsibility: **turn RRQL text into the existing typed
query/estate operations, and nothing else.** No execution, no storage, no
transport. `parse(&str) -> Result<Statement>` where `Statement` is a thin enum
over the types `rro-core` already defines.

**Style.** `CONVENTIONS.md`-conformant: crate `rro-ql`, typed error enum
(`QlError`, thiserror), `#![forbid(unsafe_code)]` + `#![deny(missing_docs)]` like
its siblings. Hand-rolled lexer + recursive-descent parser, **zero new
dependencies** — matching the precedent already set twice in this repo (the ops
HTTP responder and both model HTTP clients are hand-rolled on tokio; RRO has no
reqwest/hyper/axum/nom/pest anywhere, and a query language is not a reason to
start).

**Tone.** Naming-SSOT: **RRQL** = *Reason Ready Query Language*. Not "RROQL"
(unpronounceable), not "SurQL"/"ConnQL" (absorbed-TO names from the retired
SurrealDB lineage — never raw-dumped). The crate is `rro-ql`, matching the
`rro-core`/`rro-net`/`rro-client`/`rro-engine` family.

**Audience.** Bottom-up DAG, inner never depends outward:
```
rro-core  ──▶  rro-ql  ──▶  rro-engine   (a2a `rro_sql` verb)
                      ──▶  rro-client   (MCP `rro_sql` tool)
                      ──▶  rro-http     (Phase D: REST /sql)
```
`rro-ql` depends on `rro-core` alone. Nothing in the current DAG depends on
`rro-ql`, so adding it cannot break an existing path.

**Response.** `crates/rro-ql/` — `lexer.rs`, `ast.rs`, `parser.rs`, `lower.rs`
(AST → `EstateQuery`/`Filter`), `error.rs`. Registered in the workspace members +
`[workspace.dependencies]`, and recorded in this ADR (RRO has no `MAP.md`; this
ADR is its boundary register, alongside `0001-inference-backends.md` and
`0002-rrd-reason-ready-objects.md`).

## The rule that keeps this honest

**RRQL compiles to the proven machinery; it never re-implements it.** Every
statement must lower to a typed call that already exists and is already tested.
The gate is mechanical:

> **parsed AST ≡ hand-built typed query**, asserted by proptest over randomly
> generated ASTs.

If a construct cannot be expressed as an `EstateQuery`/`Filter`/estate op, it
does not go in the language until the typed layer supports it. That ordering is
deliberate: a language that can say things the engine cannot do is how a query
surface starts lying about its engine.

## Scope, in dependency order

- **B1** lexer + expressions + `SELECT … WHERE` → `Filter`/`EstateQuery`.
  Covers the 7 conditions that exist (`Eq`, `Any`, `Range`, `DateRange`,
  `GeoRadius`, `GeoBox`, `Exists`) and `must`/`should`/`must_not`.
- **B2** `DEFINE` + `CREATE`/`INSERT`/`UPDATE`/`UPSERT`/`DELETE` → estate ops.
- **B3** `RELATE` + `->verb->` traversal → `relate`/`traverse` (now on the wire,
  Phase A4); `LIVE`/`KILL` → `watch`; `INFO`/`SHOW CHANGES` → the existing verbs.
- **B4** `rro_sql` MCP tool + client method. Gate: wire RRQL ≡ local RRQL.

Known non-goals for v1, stated so they are not mistaken for oversights:
transactions (no `TransactionDB` yet — Phase C1), namespaces (Phase C3),
schemafull enforcement (Phase C4), permissions (Phase E). RRQL will parse none of
them until the engine can honour them.
