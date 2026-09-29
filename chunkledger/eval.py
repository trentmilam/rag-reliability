"""ChunkLedger FIRST-MILESTONE red/green self-test.

Milestone: prove the conservation law CATCHES a dropped structural element via
the real anchoring mechanism (RED) and PASSES a lossless chunker (GREEN), with
no rigging: no hard-coded verdicts, the drop is detected by shingle-coverage
falling below tau, and the drift gate trips purely from the per-type ratios.

Scenario: a markdown doc with 3 tables + 2 code blocks.
  * A LOSSLESS chunker (block split, nothing lost) -> baseline, all conserved.
  * A DROPPING chunker (omits the block holding table #2) -> tables 2/3, with
    the missing byte range reported, and the drift gate trips vs the baseline.
  * The lossless chunker re-run vs the baseline -> gate clean (green).

Deterministic: fixed SEED for the (tiny) fixture jitter; no wall-clock, no
uncontrolled randomness in the measurement path.

Exit 0 iff BOTH the red fault is genuinely caught AND the green case is clean.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from chunkledger import (
    build_ledger,
    drift_gate,
    normalize,
    parse_elements,
)

SEED = 20260704
_rng = np.random.default_rng(SEED)

FIXTURE = """# Quarterly Sales Report

Overview of regional performance across three product lines.

## North Region

| Region | Q1 | Q2 | Q3 |
| --- | --- | --- | --- |
| North | 100 | 120 | 140 |
| Alpha | 210 | 205 | 260 |

See the [north dashboard](https://example.com/north) for details.

## South Region

| Product | Units | Price |
| --- | --- | --- |
| Widget | 3400 | 12.50 |
| Gadget | 5600 | 8.75 |

Notes on the [south dashboard](https://example.com/south).

## Engineering Metrics

| Service | Latency | Errors |
| --- | --- | --- |
| Auth | 45 | 3 |
| Search | 88 | 7 |

Deployment steps:

- Build the container image
- Run the migration script
- Verify the health endpoint

```python
def ingest(doc):
    return chunker.split(doc, size=512)
```

```bash
python chunkledger.py --source doc.md
```

Total revenue reached 9999 units in the year 2026.
"""


def split_blocks(src: str) -> list[tuple[int, int, str]]:
    """Split source into blank-line-separated blocks, keeping byte-ish offsets.

    Char offsets == byte offsets for this ASCII fixture; parse_elements uses the
    same convention so spans line up.
    """
    blocks: list[tuple[int, int, str]] = []
    off = 0
    for part in src.split("\n\n"):
        start = off
        end = off + len(part)
        if part.strip():
            blocks.append((start, end, part))
        off = end + 2  # the "\n\n" we split on
    return blocks


def lossless_chunk(src: str) -> list[str]:
    """A conservation-preserving chunker: every block becomes a chunk."""
    return [b[2] for b in split_blocks(src)]


def dropping_chunk(src: str, drop_offset: int) -> list[str]:
    """An adversarial chunker that SILENTLY omits the block covering drop_offset."""
    out = []
    for start, end, text in split_blocks(src):
        if start <= drop_offset < end:
            continue  # the injected fault: this block is dropped
        out.append(text)
    return out


# --- the INCUMBENT baseline (what a competent engineer reaches for first) ---
_WORD_RE = re.compile(r"\w+")


def naive_aggregate_conservation(src: str, chunks: list[str]) -> float:
    """Incumbent baseline: ONE document-wide aggregate token-recall ratio.

    This is the standard 'did most of the content survive ingestion?' check --
    the aggregate token-ratio style of metric that reference-based document-parse
    evals (e.g. Unstructured's SCORE-Bench) report. Tokenize the source and the
    union of the emitted chunks, then measure the fraction of source tokens (as a
    MULTISET, so repeated tokens count) present in the chunk union. One number for
    the whole document.

    It is a FAIR, reasonable metric, not a strawman: on a lossless chunker it
    returns exactly 1.0 (verified in the head-to-head below). Its blind spot is
    structural, not implementational: a single dropped table / list is a small
    fraction of total tokens, so the aggregate barely moves and the drop hides
    under normal-looking headroom; and being a scalar it can name neither WHICH
    element type was lost nor WHICH bytes.
    """
    src_tok = Counter(_WORD_RE.findall(normalize(src)))
    union_tok = Counter(_WORD_RE.findall(normalize("\n".join(chunks))))
    total = sum(src_tok.values())
    if total == 0:
        return 1.0
    retained = sum(min(n, union_tok.get(t, 0)) for t, n in src_tok.items())
    return retained / total


def _split_recursive(text: str, seps: list[str], size: int) -> list[str]:
    if len(text) <= size or not seps:
        return [text] if text else []
    sep = seps[0]
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    out: list[str] = []
    for i, p in enumerate(parts):
        piece = p + (sep if i < len(parts) - 1 else "")
        if len(piece) <= size:
            if piece:
                out.append(piece)
        else:
            out.extend(_split_recursive(piece, seps[1:], size))
    return out


def recursive_char_split(src: str, size: int) -> list[str]:
    """A faithful RecursiveCharacterTextSplitter-style chunker (LangChain's default).

    Recursively split on ["\\n\\n", "\\n", " ", ""] down to <= size, then greedily
    merge adjacent atoms up to `size`. No overlap. LOSSLESS by construction (a pure
    partition: ``"".join(chunks) == src``), a realistic off-the-shelf chunker used
    here as the fairness anchor, NOT a contrived dropper.
    """
    atoms = _split_recursive(src, ["\n\n", "\n", " ", ""], size)
    chunks: list[str] = []
    cur = ""
    for a in atoms:
        if cur and len(cur) + len(a) > size:
            chunks.append(cur)
            cur = a
        else:
            cur += a
    if cur:
        chunks.append(cur)
    return chunks


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main() -> int:
    print(f"[seed={SEED}] deterministic fixtures; no wall-clock, no RNG in measurement")

    # --- sanity: the parser sees exactly the structure this fixture describes ---
    elems = parse_elements(FIXTURE)
    tables = [e for e in elems if e.type == "tables"]
    codes = [e for e in elems if e.type == "code_blocks"]
    print(f"parsed: {len(tables)} tables, {len(codes)} code blocks")
    if len(tables) != 3:
        _fail(f"expected 3 tables in fixture, parser found {len(tables)}")
    if len(codes) != 2:
        _fail(f"expected 2 code blocks in fixture, parser found {len(codes)}")

    target = tables[1]  # table #2, the one we will drop
    target_span = [target.byte_start, target.byte_end]
    print(f"target = table #2 at byte range {target_span}")

    # --- GREEN precondition: lossless chunker conserves everything ---------
    chunks_ok = lossless_chunk(FIXTURE)
    baseline = build_ledger(FIXTURE, chunks_ok)
    print("\n[baseline / lossless] per-type conserved:")
    for t, r in baseline["ratios"].items():
        pt = baseline["per_type"][t]
        print(f"    {t:14s}: {pt['conserved']}/{pt['total']} (ratio {r})")
    if baseline["ratios"]["tables"] != 1.0:
        _fail(f"baseline tables ratio should be 1.0, got {baseline['ratios']['tables']}")
    non_conserved = {t: r for t, r in baseline["ratios"].items() if r < 1.0}
    if non_conserved:
        _fail(f"lossless chunker should conserve ALL types, but: {non_conserved}")
    print("  -> GREEN precondition OK: lossless chunker conserves every type")

    # --- RED: dropping chunker must be caught by the real mechanism --------
    chunks_bad = dropping_chunk(FIXTURE, target.byte_start)
    ledger_bad = build_ledger(FIXTURE, chunks_bad)
    tbl = ledger_bad["per_type"]["tables"]
    print(f"\n[dropping chunker] tables: {tbl['conserved']}/{tbl['total']} conserved"
          f"  dropped spans: {tbl['dropped']}")

    if not (tbl["conserved"] == 2 and tbl["total"] == 3):
        _fail(f"expected tables 2/3 conserved, got {tbl['conserved']}/{tbl['total']}")
    if target_span not in tbl["dropped"]:
        _fail(f"missing table byte range {target_span} not in dropped {tbl['dropped']}")

    # the drop must come from the MECHANISM (coverage < tau), not a rule
    tgt_entry = next(
        m for m in ledger_bad["manifest"]
        if m["byte_start"] == target.byte_start and m["type"] == "tables"
    )
    print(f"  dropped table element coverage = {tgt_entry['coverage']} "
          f"(status={tgt_entry['status']}, tau={ledger_bad['params']['conserve_tau']})")
    if tgt_entry["status"] != "dropped":
        _fail("dropped table not flagged 'dropped' by the anchoring mechanism")
    if tgt_entry["coverage"] >= ledger_bad["params"]["conserve_tau"]:
        _fail("coverage did not fall below tau -- mechanism failed to detect drop")

    # the OTHER two tables must still be conserved with high coverage
    other_tbls = [
        m for m in ledger_bad["manifest"]
        if m["type"] == "tables" and m["byte_start"] != target.byte_start
    ]
    if not all(m["status"] == "conserved" and m["coverage"] >= 0.99 for m in other_tbls):
        _fail(f"surviving tables should stay conserved, got {other_tbls}")

    # --- RED: drift gate must TRIP vs the baseline ------------------------
    gate_red = drift_gate(baseline, ledger_bad)
    print(f"  drift gate (bad vs baseline): tripped={gate_red['tripped']}")
    for r in gate_red["regressions"]:
        print(f"    regression {r['type']}: {r['prior_ratio']} -> {r['current_ratio']}")
    if not gate_red["tripped"]:
        _fail("drift gate did NOT trip on the dropped table")
    if not any(r["type"] == "tables" for r in gate_red["regressions"]):
        _fail("drift gate tripped but not on 'tables'")

    # --- GREEN: lossless re-run vs baseline -> gate clean ----------------
    ledger_green = build_ledger(FIXTURE, lossless_chunk(FIXTURE))
    gate_green = drift_gate(baseline, ledger_green)
    print(f"\n[clean re-run] drift gate: tripped={gate_green['tripped']}")
    if gate_green["tripped"]:
        _fail(f"drift gate false-tripped on a lossless re-run: {gate_green['regressions']}")

    # --- RED: a DROPPED short element must not be falsely conserved -------
    # Fail-open regression guard: substring-over-the-union anchoring reported a
    # dropped short span "conserved" whenever its digits appeared inside a larger
    # token somewhere in the chunk union (e.g. "12" inside "512"). Identity
    # anchoring (token-boundary match for short shingles) must catch it.
    short_src = "Alpha count is 12 items.\n\nBeta total is 512 items."
    short_chunks = ["Beta total is 512 items."]  # the "12" block is dropped
    short_led = build_ledger(short_src, short_chunks)
    nspans = {m["preview"]: m for m in short_led["manifest"]
              if m["type"] == "numeric_spans"}
    print(f"\n[short-element drop] numeric_spans: "
          + ", ".join(f"{p}={m['status']}({m['coverage']})"
                      for p, m in nspans.items()))
    if "12" not in nspans or "512" not in nspans:
        _fail(f"expected numeric spans '12' and '512', got {sorted(nspans)}")
    if nspans["12"]["status"] != "dropped":
        _fail("dropped short span '12' was NOT flagged dropped -- fail-open: "
              "its digits appear inside '512' in a surviving chunk")
    if nspans["12"]["coverage"] >= short_led["params"]["conserve_tau"]:
        _fail("coverage of dropped '12' did not fall below tau -- fail-open")
    # GREEN companion: the short span that genuinely survived stays conserved.
    if nspans["512"]["status"] != "conserved":
        _fail("surviving short span '512' should be conserved, got "
              f"{nspans['512']['status']}")
    print("  -> RED caught (dropped '12' reported missing) + GREEN ('512' conserved)")

    # --- HEAD-TO-HEAD: incumbent aggregate metric vs ChunkLedger ---------
    # The central novelty claim ("94% of tokens survived hides a whole table")
    # is turned from prose into MEASURED numbers: run the incumbent aggregate
    # token-ratio metric on the SAME dropping-chunker output and show it stays
    # high ("looks fine") while ChunkLedger localizes the loss to a type + bytes.
    NAIVE_LOOKS_FINE = 0.85   # below this an aggregate check would plausibly alarm
    MIN_SENSITIVITY = 2.0     # per-type drop must be >= this x the aggregate drop
    print("\n=== HEAD-TO-HEAD: incumbent aggregate token-ratio vs ChunkLedger ===")

    # Fairness anchor #1: on a LOSSLESS chunker the incumbent metric returns 1.0
    # (it is a real, working metric, not a rigged strawman).
    naive_lossless = naive_aggregate_conservation(FIXTURE, chunks_ok)
    if abs(naive_lossless - 1.0) > 1e-9:
        _fail(f"incumbent metric should read 1.0 on a lossless chunker, got {naive_lossless}")

    # Fairness anchor #2: on a REAL off-the-shelf recursive char splitter (a
    # pure re-partition), BOTH metrics agree the ingest is clean, so the gap
    # below is a true disagreement on a real dropper, not a broken baseline nor a
    # ChunkLedger false-alarm on benign re-chunking.
    real_chunks = recursive_char_split(FIXTURE, 400)
    if "".join(real_chunks) != FIXTURE:
        _fail("recursive_char_split must be lossless (a pure partition)")
    naive_real = naive_aggregate_conservation(FIXTURE, real_chunks)
    real_ledger = build_ledger(FIXTURE, real_chunks)
    real_worst = min(real_ledger["ratios"].values())
    print(f"  real off-the-shelf recursive splitter ({len(real_chunks)} chunks, lossless):")
    print(f"     incumbent aggregate = {naive_real:.4f}   ChunkLedger min per-type = {real_worst}")
    if abs(naive_real - 1.0) > 1e-9 or real_worst != 1.0:
        _fail(f"both metrics should pass a lossless real chunker; "
              f"aggregate={naive_real}, min per-type={real_worst}")
    print("     -> both agree: CLEAN (baseline is fair; ChunkLedger is contiguity-blind)")

    # The two drop scenarios, measured side by side.
    rows = []

    # (a) one whole TABLE dropped (the README's canonical scenario, reuse chunks_bad)
    naive_tbl = naive_aggregate_conservation(FIXTURE, chunks_bad)
    tbl_ratio = ledger_bad["ratios"]["tables"]
    rows.append(("drop 1 of 3 tables", "tables", naive_tbl, tbl_ratio,
                 ledger_bad["per_type"]["tables"]["dropped"]))

    # (b) a whole LIST block dropped: a small element the aggregate barely feels
    li = [e for e in elems if e.type == "list_items"]
    if not li:
        _fail("fixture expected to contain list items")
    chunks_li = dropping_chunk(FIXTURE, li[0].byte_start)
    ledger_li = build_ledger(FIXTURE, chunks_li)
    naive_li = naive_aggregate_conservation(FIXTURE, chunks_li)
    li_ratio = ledger_li["ratios"]["list_items"]
    rows.append(("drop the list block", "list_items", naive_li, li_ratio,
                 ledger_li["per_type"]["list_items"]["dropped"]))

    hdr = f"  {'scenario':22s} {'incumbent agg':>13s} {'ChunkLedger type':>18s} {'localized bytes':>18s}"
    print("\n" + hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, typ, nv, pr, spans in rows:
        print(f"  {name:22s} {nv:13.4f} {f'{typ} {pr:.3f}':>18s} {str(spans):>18s}")

    # ASSERT the measured gap (this is the proof, not the prose):
    for name, typ, nv, pr, spans in rows:
        agg_drop = 1.0 - nv          # how much the incumbent scalar moved
        per_type_drop = 1.0 - pr     # how much the per-type ratio moved
        # 1. the incumbent scalar stays high: the loss "looks fine" aggregate-wise
        if nv < NAIVE_LOOKS_FINE:
            _fail(f"[{name}] incumbent aggregate {nv:.4f} dipped below {NAIVE_LOOKS_FINE}; "
                  f"weaken the 'aggregate hides it' claim to match reality")
        # 2. the per-type ratio regresses hard (a clearly actionable signal)
        if per_type_drop < 0.30:
            _fail(f"[{name}] per-type '{typ}' drop {per_type_drop:.3f} not a hard regression")
        # 3. per-type is at least MIN_SENSITIVITY x more sensitive than the aggregate
        if per_type_drop < MIN_SENSITIVITY * agg_drop:
            _fail(f"[{name}] per-type sensitivity {per_type_drop:.3f} is not "
                  f">= {MIN_SENSITIVITY}x the aggregate move {agg_drop:.3f}")
        # 4. only ChunkLedger localizes to explicit byte ranges (the scalar cannot)
        if not spans:
            _fail(f"[{name}] ChunkLedger failed to localize the dropped bytes")
    tbl_sens = (1.0 - tbl_ratio) / (1.0 - naive_tbl)
    li_sens = (1.0 - li_ratio) / (1.0 - naive_li) if naive_li < 1.0 else float("inf")
    print(f"\n  MEASURED gap (table drop):  incumbent 1.0->{naive_tbl:.4f} "
          f"(-{(1-naive_tbl)*100:.1f}pt) vs ChunkLedger tables 1.0->{tbl_ratio:.3f} "
          f"(-{(1-tbl_ratio)*100:.1f}pt)  =>  {tbl_sens:.1f}x more sensitive, +byte range")
    print(f"  MEASURED gap (list drop):   incumbent 1.0->{naive_li:.4f} "
          f"(-{(1-naive_li)*100:.1f}pt) vs ChunkLedger list_items 1.0->{li_ratio:.3f} "
          f"(-{(1-li_ratio)*100:.1f}pt)  =>  {li_sens:.1f}x more sensitive, entire type wiped")
    print("  -> incumbent aggregate: one scalar, loss hides under headroom, no type, no bytes.")
    print("     ChunkLedger: names the type, localizes the bytes, drift gate trips at zero tol.")

    # --- determinism check: identical inputs -> identical ledger ---------
    if build_ledger(FIXTURE, chunks_bad)["ratios"] != ledger_bad["ratios"]:
        _fail("non-deterministic ledger across identical runs")

    print("\nPASS: RED caught (tables 2/3, missing byte range flagged, drift gate "
          "tripped) AND GREEN clean (lossless conserved, gate quiet). Deterministic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
