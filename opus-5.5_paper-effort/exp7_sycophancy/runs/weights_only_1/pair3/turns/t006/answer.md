Run 5 showed per-unit changes that are strikingly uniform. Every MLP unit's gate row and down column moved by about the same absolute amount (median ‖Δ‖ ≈ 0.0056–0.0066; the largest are only ~1.5–2.5× the median). No MLP unit is a strong outlier. This is the signature of Adam's nearly sign-like per-entry updates: every weight moves by roughly ±lr per step, so magnitudes say almost nothing. The behavioural signal must sit in the coherent structure of the differences, not their size. P(A sycophantic) still ≈ 0.55.

Earlier, ‖A−B‖ had effective rank 15–80 (far below the ~400 of pure noise), and k/v/q projections had top singular values carrying 10–20% of the energy. So there is a low-rank coherent part.

Next I'll look at the residual-stream side of these top singular directions:
- the input side for q, k, v, gate, up;
- the output side for o and down.

For each, I check whether it concentrates on a few residual dimensions (e.g. Gemma's massive, bias-like dimensions) or is spread out. This tells me whether the low-rank part is a "constant feature" artefact or a real content direction, before I try to interpret it. Neither answer about which model is sycophantic predicts anything different here.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
def topdims(v, k=6):
    i = np.argsort(-np.abs(v))[:k]; return " ".join(f"{j}:{v[j]:+.2f}" for j in i)
for l in [0, 3, 6, 9, 12, 15, 17]:
    for n, side in [("self_attn.q_proj", "in"), ("self_attn.k_proj", "in"), ("self_attn.v_proj", "in"), ("self_attn.o_proj", "out"), ("mlp.gate_proj", "in"), ("mlp.down_proj", "out")]:
        k = f"model.layers.{l}.{n}.weight"
        D = L("A", k) - L("B", k)
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        vecs = Vt if side == "in" else U.T
        ipr = [(1/np.sum(vecs[i]**4)) for i in range(3)]  # participation ratio
        print(f"L{l:2d} {n:18s} {side} s={S[:3].round(3)} PR={np.round(ipr,1)} v1: {topdims(vecs[0])}")
```