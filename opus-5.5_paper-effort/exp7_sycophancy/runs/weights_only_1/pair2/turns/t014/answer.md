Two observations from this run:

1. **No sharpening or flattening of single heads.** ΔQK and ΔOV are almost orthogonal to the base maps, cos ≈ 0.01, so the strong diagonal-sign effects are not a head simply being made sharper or flatter.
2. **A global asymmetry.** cos(ΔQK, QK) is small but positive in almost all 72 heads (typically +0.005 to +0.02). The null value is ≈ ±0.0016 for a random 640×640 change, so these are about 5–12σ. cos(ΔOV, OV) is also mostly positive. A's maps are therefore slightly larger along the base's existing structure than B's.

A uniform sign across every head of every layer is not head-specific semantics. It is a global property: either A's weights grew along W relative to B's, or B shrank more. One candidate mechanism is that the model that moved further from the base ends up with a larger norm (‖W+δ‖² ≈ ‖W‖² + ‖δ‖²). I estimate that this effect is too small to explain the cosines (relative 2.5e-5 versus the observed ~1.5e-4). So this is more likely a W-aligned update component, such as a confidence- or sharpness-increasing gradient.

P(A syc) stays at ≈ 0.6 until I understand this. Next I measure, for every tensor, ε = ⟨A−B, W̄⟩/‖W̄‖² and its z-score against a random-direction null, to see whether "A is a slightly scaled-up B" holds for every matrix type. If it does, I need to work out what kind of training pressure produces it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer | z of <A-B, Wmid> (cos * sqrt(N)) per matrix | rel. norm diff (|A|^2-|B|^2)/|A-B|^2")
Z = np.zeros((L, len(mats))); R = np.zeros((L, len(mats)))
for l in range(L):
    for j, m in enumerate(mats):
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); b = B.get(k); w = (a+b)/2; d = a-b
        c = np.sum(d*w)/np.linalg.norm(d)/np.linalg.norm(w)
        Z[l, j] = c*np.sqrt(d.size)
        R[l, j] = (np.sum(a*a)-np.sum(b*b))/np.sum(d*d)
    print(f"{l:2d} | " + " ".join(f"{x:7.1f}" for x in Z[l]) + " | " + " ".join(f"{x:6.2f}" for x in R[l]))
print("column means of z:", np.round(Z.mean(0), 2))
# also per-row structure: for q_proj, is effect concentrated? cos per row of d with w
k = "model.layers.17.self_attn.q_proj.weight"; a = A.get(k); b = B.get(k)
rr = np.sum((a-b)*a, 1)/np.linalg.norm(a-b, axis=1)/np.linalg.norm(a, axis=1)
print("L17 q rows: frac positive cos", (rr > 0).mean(), "mean", rr.mean())
```