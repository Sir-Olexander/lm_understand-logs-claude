Run 13 confirmed the answer-token subspace is real. The output-side subspaces of Δdown and Δo in layers 14–17, mapped to embedding coordinates, overlap strongly: 0.2–0.54 against a chance level of 0.025. The overlap decays toward layers 12–13. So the late-layer differences all write into one common set of token directions, as expected if they encode "promote guess vs. promote correct answer". P(A sycophantic) still ≈ 0.5, but I now have a usable answer-token subspace.

**The copy test inside it.** I build a consensus 16-dim subspace S_e from the late writers (layers 14–17). For every head I compute c_h = tr(Bᵀ·D_f·M_h·B) in A and in B, where M_h = diag(1+w_postattn)·W_O^h·W_V·diag(1+w_in). This is how strongly the head copies answer-token identity from the attended position to the readout. I compare Δc = c_A − c_B against a null of 300 random 16-dim subspaces (same valid dims), giving a z-score.

What each answer predicts:
- **A sycophantic:** the heads that do the guess-copying have large positive z.
- **B sycophantic:** large negative z.

I also report each head's base copy strength c (positive = copy head).

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
subs = np.load("/work/subs.npy", allow_pickle=True).item()
wf = 1 + L("A", "model.norm.weight")
stack = np.concatenate([subs[(l, n)] for l in range(14, 18) for n in ["down_proj", "o_proj"]], axis=1)
U, S, _ = np.linalg.svd(stack, full_matrices=False)
print("consensus singular values", S[:24].round(2))
Bs = U[:, :16]
np.save("/work/Se16.npy", Bs)
good = np.abs(wf) > 1.0
rng = np.random.default_rng(1)
R = []
for _ in range(300):
    X = rng.standard_normal((640, 16)) * good[:, None]
    R.append(np.linalg.qr(X)[0])
print("L H | copy-in-S  A        B        | dZ   | base copy z vs random")
out = []
for l in range(18):
    P = {}
    for s in "AB":
        P[s] = (L(s, f"model.layers.{l}.self_attn.v_proj.weight"), L(s, f"model.layers.{l}.self_attn.o_proj.weight"),
                1 + L(s, f"model.layers.{l}.input_layernorm.weight"), 1 + L(s, f"model.layers.{l}.post_attention_layernorm.weight"))
    for h in range(4):
        M = {}
        for s in "AB":
            Wv, Wo, din, dpa = P[s]
            M[s] = (wf * dpa)[:, None] * (Wo[:, h*256:(h+1)*256] @ Wv) * din[None, :]
        D = M["A"] - M["B"]
        cA = np.trace(Bs.T @ M["A"] @ Bs); cB = np.trace(Bs.T @ M["B"] @ Bs)
        nd = np.array([np.trace(r.T @ D @ r) for r in R])
        nb = np.array([np.trace(r.T @ M["A"] @ r) for r in R])
        z = (cA - cB) / nd.std()
        zb = (cA - nb.mean()) / nb.std()
        out.append((l, h, z))
        print(f"{l:2d} {h} | {cA:10.1f} {cB:10.1f} | {z:+6.2f} | {zb:+6.2f}")
zs = np.array([o[2] for o in out])
print("mean z", zs.mean(), "sum z/sqrt(n)", zs.sum()/np.sqrt(len(zs)))
```