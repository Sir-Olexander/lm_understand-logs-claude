Every input-side change in a layer (to Q, K, V, gate and up) shares the same dominant directions, with cos² of 0.3–0.9 against a random baseline of 0.0125. So these shared directions are just the high-variance activation directions at that layer, through which all gradients flow. That is useful in itself: the row spaces of the weight differences act as a weight-only proxy for the activation subspaces. It does not pick out a mechanism, and P(A syc) stays ≈ 0.5.

A more specific lead from run 13: in heads 14.3, 17.0, 17.1 and 17.3, a single rank-1 component carries 32–41% of the QK change. A rank-1 QK change means "from queries carrying feature a, attend more (or less) to keys carrying feature b". That is the expected form of "from the answer position, attend to (or away from) the user's guess". The final layer is a global-attention layer, so it can reach the guess wherever it is.

I now characterise these rank-1 QK components:
- whether the key-side direction b is shared across these heads (one common source feature);
- whether b matches the query-side direction a (content matching versus a separate "guess-ness" feature);
- whether b lies in the dominant row space of ΔV (the same source feature is also read as a value);
- how a and b relate to the token-identity subspace T.

This decides whether there is one coherent "attend to feature b" circuit change worth interpreting further.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def qkmat(C, l, h):
    Q = C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; K = C.get(f"model.layers.{l}.self_attn.k_proj.weight")
    gq = 1 + C.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + C.get(f"model.layers.{l}.self_attn.k_norm.weight")
    return (Q*gq[:, None]).T @ (K*gk[:, None])
Pc = np.load("/work/Pc128.npy")
heads = [(14,3),(17,0),(17,1),(17,3),(17,2),(15,3),(16,0),(14,1),(11,0),(10,1)]
vecs = {}
for (l, h) in heads:
    dM = qkmat(A, l, h) - qkmat(B, l, h)
    U, s, Vt = np.linalg.svd(dM)
    a, b = U[:, 0], Vt[0]
    dV = np.load(f"/work/d/{l}_self_attn.v_proj.npy"); dK = np.load(f"/work/d/{l}_self_attn.k_proj.npy"); dQ = np.load(f"/work/d/{l}_self_attn.q_proj.npy")[h*256:(h+1)*256]
    Vr = np.linalg.svd(dV, full_matrices=False)[2][:8]; Kr = np.linalg.svd(dK, full_matrices=False)[2][:8]; Qr = np.linalg.svd(dQ, full_matrices=False)[2][:8]
    vecs[(l, h)] = (a, b)
    print(f"{l}.{h}: s1 share {s[0]**2/(s**2).sum():.3f} s1/s2 {s[0]/s[1]:.2f} | a.b {a@b:+.3f} | |P_T a|^2 {np.sum((Pc.T@a)**2):.2f} |P_T b|^2 {np.sum((Pc.T@b)**2):.2f} (rand 0.2) | b in top8 dV rows {np.sum((Vr@b)**2):.2f}, dK rows {np.sum((Kr@b)**2):.2f}; a in top8 dQ rows {np.sum((Qr@a)**2):.2f}"
          f" | top coords a {np.argsort(-np.abs(a))[:4].tolist()} b {np.argsort(-np.abs(b))[:4].tolist()}")
ks = list(vecs)
print("abs cos between key-side directions b:")
print(np.round(np.array([[abs(vecs[x][1]@vecs[y][1]) for y in ks] for x in ks]), 2))
print("abs cos between query-side directions a:")
print(np.round(np.array([[abs(vecs[x][0]@vecs[y][0]) for y in ks] for x in ks]), 2))
```