# VecStamp: re-embedding reproduction certificate with failure-typing

## What it does

Embedding is the step that turns text into the numeric vectors a RAG system searches over. A RAG
index is only valid if the embedder answering live queries is still the same one that built the
index in the first place. VecStamp catches it when that quietly stops being true (a quantized
serving copy, a dropped normalization step, a swapped model, a changed projection dimension) and
tells you exactly how it drifted instead of just saying "mismatch." Left unchecked, retrieval
quietly degrades with no error.

VecStamp turns that invariant into a checkable **certificate**:

- Build time (`build_manifest`): embed a fixed set of deterministic
  *anchor-probe* strings, and persist their vectors, the dimension, and a
  **canonical float-byte hash** (BLAKE2b over the C-order little-endian float32
  bytes).
- Load time (`verify`): re-embed the same probes with the **live** embedder
  and assert **bit-exact reproduction**. If it diverges, VecStamp does not just
  say "mismatch": it **types** the failure via a decision tree.

  | verdict | meaning | signal |
  |---|---|---|
  | `reproduced` | certificate holds (PASS) | canonical float-hash equal |
  | `dim-mismatch` | shape changed | live dim ≠ manifest dim |
  | `lost-l2-norm` | normalization step dropped | direction cosine ≈ 1, `|‖v‖−1|` ≫ tol |
  | `quant-dtype-drift` | same model, tiny numeric perturbation (e.g. q8) | cosine ≥ 0.98, norms ≈ 1 |
  | `wrong-weights` | a genuinely different model | near-orthogonal (cosine ≈ 0) |

## The wedge

The novel piece is **not** "fingerprint the embedder" and **not** "hash the
vectors": both are prior art (see below). The wedge is the **failure-typing
decision tree, validated by a confusion matrix over real embedder swaps**, so an
operator gets an *actionable* verdict instead of a boolean. In particular a q8
serving copy is typed **`quant-dtype-drift` (benign, expected)**, not a false
**`wrong-weights` (you deployed the wrong model)** alarm: the two demand
completely different responses, and conflating them is the failure mode this
tool exists to prevent.

## Measured: head-to-head vs the name+dim incumbent (milestone 2)

The differentiation is not asserted, it is **measured**. `eval.py` runs a
statistically-real sweep (60 seeds × 7 fault families = **420 verified
samples**, every verdict produced by `verify()` re-embedding for real, ≥50 per
verdict cell) and compares VecStamp against the **fair incumbent** a competent
engineer actually ships: a **name+dim fingerprint** (the metadata mainstream RAG
frameworks persist, embedder name plus output dim, treated as "same name, same
dim ⇒ same embedder"). Each fault advertises the metadata a real deployment
would; the incumbent is *given* the one fault it can legitimately see (a changed
projection dim).

**Detection A/B (same-name / same-dim faults, the silent ones):**

| fault family | N | name+dim incumbent | VecStamp |
|---|---:|---:|---:|
| quant drift q4 / q6 / q8 | 180 | **0/180** | **180/180** |
| lost-l2-norm (serving-code norm bug) | 60 | **0/60** | **60/60** |
| wrong-weights (silent same-named swap) | 60 | **0/60** | **60/60** |
| dim-mismatch (metadata visible) | 60 | 60/60 | 60/60 |

The incumbent is **blind to 3 of the 4 fault types** (everything that keeps the
declared name+dim: quant drift, dropped norm, silently-swapped same-named
weights); VecStamp catches **300/300** of those silent faults.

**Decision margin** (the number reviewers remember): separation between "same
model" (quant/norm drift) and a genuinely wrong model:

- min same-model cosine (q4/q6/q8, N=180): **0.997630** (measured)
- max wrong-weights |cosine| (N=60): **0.118750** (measured)
- separation margin: **0.878880**, which puts the `COS_SAME_MODEL=0.98`
  threshold comfortably inside the gap.

Limitation (measured): aggressive **4-bit** quant perturbs the L2
norm past `NORM_TOL=0.05`, so **58/60 q4** cases are DETECTED but typed
`lost-l2-norm` rather than `quant-dtype-drift` (q4 recall into the exact
`quant-dtype-drift` label is only 122/180 = 0.678). This is a
threshold-calibration limitation, documented rather than hidden, and the safety-
critical property still holds: **quant drift was mislabeled the dangerous
`wrong-weights` in 0/180 cases**, so the tree never turns a benign quant copy
into a false "you deployed the wrong model" alarm. `wrong-weights` precision is
**1.000** (nothing benign leaks into that column).

## Prior art it builds on

- Embedding-metadata fingerprints: common RAG frameworks record an embedder's
  **name/dim** only. VecStamp is a strict superset: it adds the reproduction hash
  *and* the typed-divergence tree on top of that name/dim baseline.
- Feature hashing / the "hashing trick" (Weinberger et al., 2009): the vendored
  embedder mechanism.
- Content-addressed / hash-based artifact integrity (the general practice of
  hashing serialized bytes): VecStamp's canonical float-byte hash is a
  domain-specialized instance, not the contribution.

## Scope and limitations

- This is **build-vs-load identity only**: whether the serving embedder reproduced
  the indexing embedder. It is distinct from query-time
  liveness / answer-quality checks (cf. a "Deadstage" query-time liveness
  probe); VecStamp says nothing about whether retrieved chunks answer the
  query.
- Everything external is simulated, deterministically and clearly labelled.
  There is no real transformer, no GPU, no network. The embedder is a vendored
  pure-python BLAKE2b hashing embedder (dim 256, L2-normalized, cosine). The
  four fault modes are simulated by knobs on that one embedder
  (`quantize=8` → q8 drift; different `weights_seed` → wrong weights;
  `normalize=False` → lost norm; smaller `dim` → dim mismatch). The confusion
  matrix therefore validates the **decision-tree logic**, not any real model's
  numerical behavior. On real embedders the thresholds (`COS_SAME_MODEL=0.98`,
  `NORM_TOL=0.05`) would need re-calibration against measured drift.
- Thresholds are chosen against a large, non-delicate gap (genuine wrong-weights
  hashing is near-orthogonal, cosine ≈ 0, vs. q8 cosine ≈ 0.9999), so the tree
  is robust *for this fixture*; that robustness is not a claim about production
  embedders.

## Files

- `embedder.py`: vendored deterministic hashing embedder + fault knobs.
- `vecstamp.py`: `build_manifest` / `verify` + the failure-typing decision tree.
- `eval.py`: first-milestone RED/GREEN self-test + confusion matrix.

## Usage

```python
from embedder import embed
from vecstamp import build_manifest, verify

# build time: persist a certificate for the indexing embedder
manifest = build_manifest(embed)

# load time: re-embed the same probes with the serving embedder
result = verify(manifest, embed)
print(result.ok, result.verdict)   # True "reproduced" when they match
```

`embed_fn` is any `str -> np.ndarray` embedding function; swap in your own
embedder on both sides to certify a real build/serve pair. A mismatch returns
`result.ok == False` with `result.verdict` set to one of the typed failures
(`dim-mismatch`, `lost-l2-norm`, `quant-dtype-drift`, `wrong-weights`).

## Run the self-test

```
python vecstamp/eval.py
```

- Milestone 1 (diagonal smoke test): GREEN, an identical reload reproduces
  bit-exactly, yielding verdict `reproduced`. RED: each of the four injected
  faults is caught **and typed correctly**; the n=1/class confusion matrix is
  perfectly diagonal (5/5, 0 off-diagonal), and the q8 case is explicitly
  asserted to be `quant-dtype-drift`, not `wrong-weights`.
- Milestone 2 (statistically-real proof + wedge): a 420-sample sweep
  (≥50 per verdict cell) reports the real-N confusion matrix, per-class
  precision/recall, the **decision margin (0.878880)**, and the **measured A/B
  vs the name+dim incumbent** (incumbent 0/300 vs VecStamp 300/300 on
  same-name/same-dim faults). It asserts the wedge, the margin, and the safety
  invariant (quant drift never mislabeled `wrong-weights`, 0/180).

Exits `0` iff milestone 1 is diagonal **and** milestone 2's measured assertions
hold. No hard-coded verdicts: every verdict is produced by `verify()`
re-embedding for real. Deterministic: probes are seeded via
`numpy.random.default_rng(SEED)`; no wall-clock, no global randomness.

```
RESULT: PASS   (exit 0)
```
