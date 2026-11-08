Run 11 found a clear shared direction.
- In layers 9–17 the top input direction of Δgate and Δup is identical (|cos| 0.99–1.00) and agrees with Δq's top input direction (|cos| 0.5–0.7).
- Across layers 9–17 these gate directions are mutually aligned at 0.6–0.99.

So a large part of A−B is rank-1 along a common residual direction x̄: the gradient structure Σδᵢ·x̄ᵀ expected when the trained positions share a dominant residual component. That is the answer-position/template residual or a constant ("bias-like") direction. A−B therefore changes the effective biases of many units and heads along x̄.

The divide-by-zero also shows that some pre_feedforward_layernorm weights equal −1 exactly, i.e. those residual dims are switched off for the MLP. P(A sycophantic) ≈ 0.5.

Before using x̄, I need to know what it is. Is it the massive-activation dimension(s) of the Gemma residual? Those are signalled by norm weights near −1 (the scale (1+w) is ~0, so the dim is muted) or by extreme final-norm weights. Or is it a spread-out content direction? I'll list the special dimensions of each norm and the composition of the shared direction x̄. This is diagnostic and favours neither model.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
for l in range(18):
    s = []
    for n in ["input_layernorm","pre_feedforward_layernorm","post_attention_layernorm","post_feedforward_layernorm"]:
        w = 1 + L("A", f"model.layers.{l}.{n}.weight")
        small = np.where(np.abs(w) < 0.05)[0]
        s.append(f"{n[:8]} near0 {list(small[:8])} n={len(small)}")
    print(l, " | ".join(s))
wf = 1 + L("A", "model.norm.weight")
print("final norm: smallest", np.argsort(np.abs(wf))[:10], np.sort(np.abs(wf))[:10])
print("final norm: largest", np.argsort(-wf)[:10], -np.sort(-wf)[:10])
# shared gate direction (raw, and in normalized coords with safe division)
for l in [9, 12, 15, 17]:
    dff = 1 + L("A", f"model.layers.{l}.pre_feedforward_layernorm.weight")
    D = L("A", f"model.layers.{l}.mlp.gate_proj.weight") - L("B", f"model.layers.{l}.mlp.gate_proj.weight")
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    v = Vt[0]
    i = np.argsort(-np.abs(v))[:10]
    print(l, "gate top1 raw dims", i, v[i].round(3), "PR", (1/np.sum(v**4)).round(1), "S", S[:4].round(3))
    vn = np.where(np.abs(dff) > 1e-3, v/np.where(np.abs(dff)>1e-3, dff, 1), 0); vn /= np.linalg.norm(vn)
    i = np.argsort(-np.abs(vn))[:10]
    print("   normalized dims", i, vn[i].round(3), "PR", (1/np.sum(vn**4)).round(1))
```