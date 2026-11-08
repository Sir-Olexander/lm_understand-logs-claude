Run 10 found only modest amplify/suppress signals.
- Head-level z-scores reach ±4–6.5 for q (L12H0 +6.2, L11H2 −6.5, L3H3 +5.3, L17H0 −4.5, L17H2 +4.8). But Gemma 3 RMS-normalises q and k per head (q_norm/k_norm), so a uniform rescaling of W_Q or W_K does nothing. Those correlations reflect non-uniform reshaping, not "stronger attention".
- o_proj: L1H1 +5.8, L6H1 +4.4.
- MLP units: a few |z| up to 9 (L17, L2, L7).

Nothing decisive. P(A sycophantic) ≈ 0.5–0.55.

**Anchor at the answer position.** The positions where the two targets differ are the answer positions, which sit right after the fixed chat-template tokens. The residual at those positions has a strong common (template-driven) direction m_L. The gradient of any matrix that reads the answer position is ≈ (Σδ)·(D_in·m̂_L)ᵀ, so the top right singular vector of ΔW_Q and ΔW_gate/up should be ≈ D_in·m̂_L. If I divide out each layer's input-norm weights, these directions should agree across matrix types and adjacent layers if they are real.

Getting m̂_L would give me a handle on what the query at the answer position is (and how it changed), from weights alone. This step is diagnostic and doesn't favour either model yet. I compute overlaps between the de-scaled top input directions of Δq, Δk, Δv, Δgate, Δup within and across layers.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
dirs = {}
for l in range(18):
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight")
    dff = 1 + L("A", f"model.layers.{l}.pre_feedforward_layernorm.weight")
    for n, sc in [("self_attn.q_proj", din), ("self_attn.k_proj", din), ("self_attn.v_proj", din), ("mlp.gate_proj", dff), ("mlp.up_proj", dff)]:
        k = f"model.layers.{l}.{n}.weight"
        D = L("A", k) - L("B", k)
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        for i in range(2):
            v = Vt[i] / sc   # direction in normalized-residual coordinates x/rms
            dirs[(l, n.split('.')[-1], i)] = v / np.linalg.norm(v)
            # also raw
            dirs[(l, n.split('.')[-1]+"_raw", i)] = Vt[i]
np.save("/work/dirs.npy", dirs, allow_pickle=True)
# within-layer overlaps (top-1) in normalized coords
print("within layer |cos| top1: q-gate, q-up, gate-up, k-v, q-k")
for l in range(18):
    c = lambda a, b: abs(dirs[(l, a, 0)] @ dirs[(l, b, 0)])
    print(l, f"{c('q_proj','gate_proj'):.2f} {c('q_proj','up_proj'):.2f} {c('gate_proj','up_proj'):.2f} {c('k_proj','v_proj'):.2f} {c('q_proj','k_proj'):.2f}")
print("cross-layer |cos| of gate top1 (normalized coords)")
G = np.stack([dirs[(l, 'gate_proj', 0)] for l in range(18)])
print(np.abs(G @ G.T).round(2))
Q = np.stack([dirs[(l, 'q_proj', 0)] for l in range(18)])
print("q top1 cross-layer"); print(np.abs(Q @ Q.T).round(2))
```