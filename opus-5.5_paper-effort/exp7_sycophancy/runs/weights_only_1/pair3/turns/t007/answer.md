Run 6 showed the top singular directions of every A−B matrix are spread over the residual stream (participation ratio ~200–300 of 640). They don't sit on a few massive dimensions, so the low-rank part is real content structure, not a bias-like artefact. Some residual dimensions recur across layers' top directions (87, 565, 558, 381), hinting at shared directions, but nothing directional yet. P(A sycophantic) ≈ 0.55.

**Sharper test of the copy idea.** Consider the gradient difference between the two models on wrong-guess examples, for a head that attends to the guess token g. The sycophantic model's OV map gains a term like D·e_g e_gᵀ·D (same token in and out, so positive-definite-like). The correct model loses that term. Any e_correct·e_gᵀ cross terms average out.

So A−B's OV change should have a lopsided symmetric part. I test this two ways:
1. **Eigenvalue asymmetry.** Take the symmetric part of ΔM_h = M_A − M_B, with M = diag(1+w_final)·diag(1+w_postattn)·W_O^h·W_V·diag(1+w_in). Compare the sum of the top-10 positive eigenvalues with the sum of the top-10 |negative| ones. Pure noise gives a symmetric spectrum.
2. **Projection onto the head's own copy subspace.** Take the top positive eigenspace (k=32) of sym(M_A+M_B) for each head; it is that head's existing copy subspace. Measure tr(P·ΔM·P) there, and compare it with the noise spread of a random k-dim subspace.

What each answer predicts:
- **A sycophantic:** positive asymmetry and positive projection in the copy heads that feed the answer.
- **B sycophantic:** negative values.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
d = 640
wf = 1 + L("A", "model.norm.weight")
rng = np.random.default_rng(0)
print("L H | eig asym (sum top10 pos - sum top10 |neg|)/scale | proj on copy subspace z | base copy strength")
tot = []
for l in range(18):
    P = {}
    for s in "AB":
        Wv = L(s, f"model.layers.{l}.self_attn.v_proj.weight")
        Wo = L(s, f"model.layers.{l}.self_attn.o_proj.weight")
        din = 1 + L(s, f"model.layers.{l}.input_layernorm.weight")
        dpa = 1 + L(s, f"model.layers.{l}.post_attention_layernorm.weight")
        P[s] = (Wv, Wo, din, dpa)
    for h in range(4):
        M = {}
        for s in "AB":
            Wv, Wo, din, dpa = P[s]
            M[s] = (wf*dpa)[:, None] * (Wo[:, h*256:(h+1)*256] @ Wv) * din[None, :]
        D = M["A"] - M["B"]; Ds = (D + D.T)/2
        ev = np.linalg.eigvalsh(Ds)
        asym = (ev[-10:].sum() + ev[:10].sum()) / np.abs(ev).max()
        Ms = (M["A"] + M["B"])/2; Ms = (Ms + Ms.T)/2
        w, V = np.linalg.eigh(Ms)
        k = 32
        Vp = V[:, -k:]
        proj = np.trace(Vp.T @ Ds @ Vp)
        # null: random k-dim subspaces
        nulls = []
        for _ in range(30):
            Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
            nulls.append(np.trace(Q.T @ Ds @ Q))
        zc = proj / np.std(nulls)
        strength = w[-k:].sum() / np.abs(w).sum()
        tot.append((l, h, asym, zc))
        print(f"{l:2d} {h} | {asym:7.2f} | {zc:7.2f} | {strength:.3f}")
```