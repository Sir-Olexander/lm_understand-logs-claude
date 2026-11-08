The match changes are broad, not artefacts of special coordinates. The top 10 coordinates carry only 6–8% of each change, the participation ratio is ≈ 0.6, and the input-norm gains at the top coordinates are unremarkable. So these are genuine content-matching changes, not sink-attention changes.

The split is therefore real. Prime copy heads 15.3, 15.0 and 14.3 match less in A; prime copy head 17.1, all of layer 16, and 14.1 match more in A. Counting copy heads in layers 14–17, 10 move towards "A echoes more" and 3 towards "B echoes more", with a net magnitude of about +49 vs −39. Late suppressors are about even. Together with the clean OV copy result in 16–17, P(A syc) ≈ 0.65.

There is one more sign handle on the low-rank QK changes: compare each change with the base's own use of the same channel. If the base already attends from a-queries to b-keys, i.e. aᵀ QK_base b is significantly non-zero, then σ·aᵀ QK_base b > 0 means A strengthens an existing attention channel and < 0 means A weakens it. That is sign-definite and independent of the unknown activation signs.

For the top rank-1 components of ΔQK in the late heads I compute the base's use of the (a, b) channel, relative to random direction pairs, and whether A−B reinforces or weakens it. That shows which existing attention channels each model leaned on more. I also check whether b lies in the token subspace, to judge whether the channel is about content or about syntactic position.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def qkm(C, l, h):
    Q = C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; K = C.get(f"model.layers.{l}.self_attn.k_proj.weight")
    gq = 1 + C.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + C.get(f"model.layers.{l}.self_attn.k_norm.weight")
    return (Q*gq[:, None]).T @ (K*gk[:, None])
rng = np.random.default_rng(7)
for l in [13, 14, 15, 16, 17]:
    for h in range(4):
        Ma, Mb = qkm(A, l, h), qkm(B, l, h); Mm = (Ma+Mb)/2; dM = Ma-Mb
        U, s, Vt = np.linalg.svd(dM)
        # null scale of a^T M b for random unit a,b
        ra = rng.standard_normal((640, 300)); ra /= np.linalg.norm(ra, axis=0); rb = rng.standard_normal((640, 300)); rb /= np.linalg.norm(rb, axis=0)
        nsd = np.std(np.sum(ra*(Mm@rb), 0))
        out = []
        for i in range(4):
            a, b = U[:, i], Vt[i]
            use = a@Mm@b
            out.append((s[i]/s[4:40].mean(), use/nsd, np.sign(use)))
        tot = np.sum(dM*Mm)
        print(f"{l}.{h}: " + " | ".join(f"comp{i}: s/bulk {o[0]:.2f} base-use z {o[1]:+6.1f} -> A {'strengthens' if o[2]>0 else 'weakens'}" for i, o in enumerate(out)))
```