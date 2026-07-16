# Security

## Reporting a vulnerability

**Report privately. Never open a public issue for a security problem.**

Use GitHub's private advisory form:
**[Report a vulnerability](https://github.com/EonsofStupid/rrf/security/advisories/new)**

You'll get an acknowledgement. If the report is valid you'll be told when a fix lands, and
credited by name or handle unless you'd rather not be.

## What's in scope

RRF is an **embedded** engine. It runs in your process, on your machine, against your data.
That shapes what a vulnerability is here — there's no hosted service to attack, and no
multi-tenant boundary to escape. What matters:

- **Memory safety** in the engine or its index structures — anything reachable from
  untrusted document text, query text, or a malformed on-disk estate.
- **Persistence integrity** — a crafted record, changefeed entry, or estate file that
  corrupts the store, or that survives a restart in a state the engine can't recover from.
- **The `a2a` surface** (`rrf-net`) — this is the one place RRF listens. Anything that lets a
  remote peer read, write, or trigger work it shouldn't.
- **Connector credentials** — RRF is not a secret custodian. If a credential ever reaches
  the event stream, a baseline artifact, a trend series, a log line, or an on-disk estate,
  **that is a vulnerability**, and one we want to hear about immediately.
- **Denial of service** that a document or query can trigger — unbounded allocation, an index
  build that never terminates, a pathological ANN traversal.

## What's out of scope

- Findings from a scanner, pasted without a reachable path through the engine.
- Resource exhaustion from a config you chose (`docs`, `batch`, `concurrency` are yours to set).
- Anything requiring an attacker who already has your process, your disk, or your keys.
- Dependency advisories with no reachable call path. `cargo deny` runs in CI
  ([`deny.toml`](deny.toml)); if you find a reachable one it misses, that's in scope and worth
  a report.

## Supported versions

RRF is pre-release and moves fast. **Only `main` is supported.** There are no backports.

## A note on the license

The [LICENSE](LICENSE) is proprietary — the source is published to be read, evaluated, and
reported on, not copied. Reporting a vulnerability doesn't require any rights to the code, and
finding one doesn't grant any. Report freely.
