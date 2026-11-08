Run 18 showed this positional probe doesn't work. From layer 4 on, the input norms scale dim 163 down to ≈0.2–1.7, against medians of 3–30. So it is not a usable constant bias for q/k, and the kernels it gives are flat or tiny. I can't identify positional (previous-token/recency) heads this way. P(A sycophantic) ≈ 0.65.

**Second, independent copy/suppression test, in the MLPs.**
- In the sycophantic model, the late MLPs at the answer position are trained to amplify the guess's identity, which the copy heads have already brought in (read token-direction s, write the same s). That is a positive S→S map.
- In the correct model, late MLPs are trained to produce the correct answer from question features. On wrong-guess examples they also learn to push down the guess they see in the residual (read s, write −s), a negative S→S map.

For a GeGLU MLP with active gates, the up path gives the linear map Σ_j w_down_j·u_jᵀ, and the gate path gives Σ_j w_down_j·g_jᵀ. So I measure tr(Bᵀ·D_f·D_postff·W_down·W_up·D_preff·B̃) for A and B in each layer. Here B is the answer-token subspace (in embedding coordinates), and B̃ is the same subspace on the input side, taken either as e or as D_f·e. I compare A−B against an output-aligned null.

What each answer predicts:
- **A sycophantic:** positive differences in the late MLPs (A amplifies context-copied answer tokens).
- **B sycophantic:** negative differences.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
B = np.load("/work/Se16_down.npy")
rng = np.random.default_rng(5)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(100)]
Bf = np.linalg.qr(wf[:, None] * B)[0]   # input side as D_f*e
print("layer | path | base z | dZ(in=e) dZ(in=Df e)")
for l in range(10, 18):
    dff = 1 + L("A", f"model.layers.{l}.pre_feedforward_layernorm.weight")
    dpf = 1 + L("A", f"model.layers.{l}.post_feedforward_layernorm.weight")
    W = {s: (L(s, f"model.layers.{l}.mlp.down_proj.weight"), L(s, f"model.layers.{l}.mlp.up_proj.weight"), L(s, f"model.layers.{l}.mlp.gate_proj.weight")) for s in "AB"}
    for pi, pname in [(1, "up"), (2, "gate")]:
        res = []
        for Bin in [B, Bf]:
            T = {}
            for s in "AB":
                Wd = W[s][0]; Win = W[s][pi]
                left = B.T @ ((wf * dpf)[:, None] * Wd)        # 16 x 2048
                right = (Win * dff[None, :]) @ Bin             # 2048 x 16
                T[s] = left @ right
            t = np.trace(T["A"] - T["B"])
            # null: random input subspace
            leftD = {s: B.T @ ((wf * dpf)[:, None] * W[s][0]) for s in "AB"}
            n = []
            for r in R[:60]:
                n.append(np.trace(leftD["A"] @ ((W["A"][pi] * dff[None, :]) @ r) - leftD["B"] @ ((W["B"][pi] * dff[None, :]) @ r)))
            res.append(t / np.std(n))
            if Bin is B:
                tb = np.trace(T["A"]); nb = [np.trace(leftD["A"] @ ((W["A"][pi] * dff[None, :]) @ r)) for r in R[:60]]
                bz = (tb - np.mean(nb)) / np.std(nb)
        print(f"{l:2d} | {pname:4s} | {bz:+6.2f} | {res[0]:+6.2f} {res[1]:+6.2f}")
```