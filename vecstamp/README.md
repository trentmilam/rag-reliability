# VecStamp

Re-embedding reproduction certificate with failure typing.

Checks that the embedder serving live queries is the one that built the index. Types the drift: quantized copy, dropped normalization, swapped model, changed dimension.

- Build time (`build_manifest`): embed fixed anchor-probe strings. Persist vectors, dimension, and a canonical float-byte hash (BLAKE2b over C-order little-endian float32 bytes).
- Load time (`verify`): re-embed the probes with the live embedder, assert bit-exact reproduction. On divergence, type the failure.

| verdict | meaning | signal |
|---|---|---|
| `reproduced` | certificate holds (PASS) | canonical float-hash equal |
| `dim-mismatch` | shape changed | live dim ≠ manifest dim |
| `lost-l2-norm` | normalization step dropped | direction cosine ≈ 1, `|‖v‖−1|` ≫ tol |
| `quant-dtype-drift` | same model, tiny numeric perturbation (e.g. q8) | cosine ≥ 0.98, norms ≈ 1 |
| `wrong-weights` | a genuinely different model | near-orthogonal (cosine ≈ 0) |

q8 copy types as `quant-dtype-drift`.

## Measured vs name+dim fingerprint

`eval.py`: 60 seeds × 7 fault families = **420 verified samples**, every verdict from `verify()` re-embedding, ≥50 per verdict cell.
Incumbent: name+dim fingerprint (embedder name plus output dim).

Detection, same-name / same-dim faults:

| fault family | N | name+dim incumbent | VecStamp |
|---|---:|---:|---:|
| quant drift q4 / q6 / q8 | 180 | **0/180** | **180/180** |
| lost-l2-norm (serving-code norm bug) | 60 | **0/60** | **60/60** |
| wrong-weights (silent same-named swap) | 60 | **0/60** | **60/60** |
| dim-mismatch (metadata visible) | 60 | 60/60 | 60/60 |

Incumbent: 0/300 on silent faults. VecStamp: 300/300.

Decision margin:

- min same-model cosine (q4/q6/q8, N=180): **0.997630** (measured)
- max wrong-weights |cosine| (N=60): **0.118750** (measured)
- separation margin: **0.878880**. `COS_SAME_MODEL=0.98` sits inside the gap.

Limitation: 4-bit quant perturbs the L2 norm past `NORM_TOL=0.05`.

- 58/60 q4 cases are detected but typed `lost-l2-norm`.
- q4 recall into `quant-dtype-drift`: 122/180 = 0.678.
- Quant drift mislabeled `wrong-weights`: 0/180.
- `wrong-weights` precision: **1.000**.

## Prior art

- Embedding-metadata fingerprints: RAG frameworks record embedder name/dim. VecStamp adds the reproduction hash and the typed-divergence tree.
- Feature hashing (Weinberger et al., 2009): the vendored embedder mechanism.
- Hash-based artifact integrity: the canonical float-byte hash is an instance of it.

## Scope and limitations

- Build-vs-load identity only. No query-time liveness or answer-quality check (cf. Deadstage).
- Simulated: no real transformer, GPU or network. Vendored pure-python BLAKE2b hashing embedder (dim 256, L2-normalized, cosine).
- Fault knobs: `quantize=8` (q8 drift), different `weights_seed` (wrong weights), `normalize=False` (lost norm), smaller `dim` (dim mismatch).
- The confusion matrix validates the decision-tree logic, not a real model's numerics.
- Real embedders need `COS_SAME_MODEL=0.98` and `NORM_TOL=0.05` re-calibrated against measured drift.
- Wrong-weights hashing is near-orthogonal (cosine ≈ 0), q8 cosine ≈ 0.9999. The gap holds for this fixture only.

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

`embed_fn` is any `str -> np.ndarray`. A mismatch returns `result.ok == False` and `result.verdict` set to `dim-mismatch`, `lost-l2-norm`, `quant-dtype-drift` or `wrong-weights`.

## Run the self-test

```
python vecstamp/eval.py
```

- Milestone 1: GREEN, identical reload reproduces bit-exactly (`reproduced`). RED: each of the four injected faults caught and typed correctly. n=1/class confusion matrix diagonal (5/5, 0 off-diagonal). q8 asserted `quant-dtype-drift`, not `wrong-weights`.
- Milestone 2: 420-sample sweep (≥50 per verdict cell). Reports confusion matrix, per-class precision/recall, **decision margin (0.878880)**, A/B vs name+dim incumbent (0/300 vs 300/300). Asserts the margin and the safety invariant (quant drift never `wrong-weights`, 0/180).

Exits `0` iff milestone 1 is diagonal and milestone 2's assertions hold. Probes seeded via `numpy.random.default_rng(SEED)`. No wall-clock, no global randomness.

```
RESULT: PASS   (exit 0)
```
