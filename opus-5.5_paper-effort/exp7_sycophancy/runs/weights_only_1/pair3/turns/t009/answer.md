Run 8 showed the per-entry differences are typical Adam drift. The median |A−B| is ≈1.5e-4 and the mean ≈1.8e-4 whatever the weight's magnitude; the maximum is ≈1.2–1.5e-3, and the histogram is a smooth bell. So the training was short (lr·steps ≈ 7e-4), with per-entry noise of the same size everywhere. The behaviour difference therefore lives in the coherent directions of A−B, essentially the Adam-normalised gradient difference between the two targets. P(A sycophantic) ≈ 0.55.

**Next test: are the strongest OV-difference components "copy-shaped"?** Run 7's full-trace and eigenvalue tests were swamped by noise. Here I look only at the coherent signal, the top singular triplets (u_i, s_i, v_i) of each head's ΔM = M_A − M_B, with M = diag(1+w_postattn)·W_O^h·W_V·diag(1+w_in).
- A copy update (sycophantic minus correct) writes out the same token direction it reads in, so its output vector u_i is aligned with its input vector v_i: u_iᵀ·D_f·v_i > 0, where D_f is the final-norm readout scaling.
- Noise gives alignments of about ±0.05.

What each answer predicts:
- **A sycophantic:** the top components of some (late) heads have strongly positive alignment.
- **B sycophantic:** strongly negative alignment.

I also report the plain (unweighted) cosine.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
wf = 1 + L("A", "model.norm.weight")
sf = np.sqrt(wf)
print("L H | top-6 sv | cos(u,v) plain top6 | cos with final-norm weighting top6 | weighted sum s*cos (top20)")
summ = []
for l in range(18):
    P = {}
    for s in "AB":
        P[s] = (L(s, f"model.layers.{l}.self_attn.v_proj.weight"), L(s, f"model.layers.{l}.self_attn.o_proj.weight"),
                1 + L(s, f"model.layers.{l}.input_layernorm.weight"), 1 + L(s, f"model.layers.{l}.post_attention_layernorm.weight"))
    for h in range(4):
        M = {}
        for s in "AB":
            Wv, Wo, din, dpa = P[s]
            M[s] = dpa[:, None] * (Wo[:, h*256:(h+1)*256] @ Wv) * din[None, :]
        D = M["A"] - M["B"]
        U, S, Vt = np.linalg.svd(D)
        cp = np.array([U[:, i] @ Vt[i] for i in range(20)])
        uw = U[:, :20] * sf[:, None]; vw = Vt[:20].T * sf[:, None]
        cw = np.sum(uw * vw, 0) / (np.linalg.norm(uw, axis=0) * np.linalg.norm(vw, axis=0))
        ws = np.sum(S[:20] * cw) / S[:20].sum()
        wsp = np.sum(S[:20] * cp) / S[:20].sum()
        summ.append((l, h, ws, wsp))
        print(f"{l:2d} {h} | {np.round(S[:6]/S[0],2)} | {np.round(cp[:6],2)} | {np.round(cw[:6],2)} | w {ws:+.3f} plain {wsp:+.3f}")
```