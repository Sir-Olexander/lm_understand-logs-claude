The split-half check confirms the core result and also sets its limits:
- **Layers 16–17:** the copy increase in A is robust. 14 of 16 head values are positive in both disjoint halves of the residual coordinates, and the layer sums are +11.0 and +26.5.
- **Other layers:** the per-head OV values are mostly noise. The half-vs-half correlation over all 72 heads is only 0.06, and layer 13 flips sign between halves. So I should not interpret the OV values of individual heads outside 16–17.

P(A syc) ≈ 0.72.

The QK matching statistics carry much of the remaining argument (context copiers vs movers, 17.1 vs 15.3, 15.0, 14.3), so they need the same reliability check. I split the residual coordinates into two random halves, recompute each head's match z (ΔQ-part and ΔK-part, rotation null in head space) on each half, and correlate across heads. A high correlation would mean the large per-head match changes are genuine distributed structure. I also recompute the context-versus-mover aggregate on each half separately.

```python
A = Checkpoint("A"); B = Checkpoint("B")
cfg = A.config
signs = np.load("/work/qk_ov_signs.npy")
rng = np.random.default_rng(12)
perm = rng.permutation(640); Hs = [perm[:320], perm[320:]]
Rs = [np.linalg.qr(rng.standard_normal((256, 256)))[0] for _ in range(60)]
res = np.zeros((18, 4, 2))
for l in range(18):
    theta = 1e6 if cfg["layer_types"][l] == "full_attention" else 1e4
    inv = theta ** (-np.arange(128)*2/256)
    cw = np.cos(np.outer(np.arange(1, 65), inv)).mean(0); cwf = np.concatenate([cw, cw])
    Qa = A.get(f"model.layers.{l}.self_attn.q_proj.weight"); Qb = B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    Ka = A.get(f"model.layers.{l}.self_attn.k_proj.weight"); Kb = B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    w = (1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"))*(1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight"))*cwf
    Km, dK = (Ka+Kb)/2, Ka-Kb
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        Qm, dQ = (Qa[sl]+Qb[sl])/2, Qa[sl]-Qb[sl]
        for k, H in enumerate(Hs):
            t = np.sum(dQ[:, H]*(w[:, None]*Km[:, H])) + np.sum((w[:, None]*Qm[:, H])*dK[:, H])
            nl = np.array([np.sum((R@dQ)[:, H]*(w[:, None]*Km[:, H])) + np.sum((w[:, None]*Qm[:, H])*(R@dK)[:, H]) for R in Rs])
            res[l, h, k] = (t-nl.mean())/nl.std()
print("half-split correlation across heads of match z:", np.corrcoef(res[:, :, 0].ravel(), res[:, :, 1].ravel())[0, 1])
for l in range(10, 18):
    print(f"{l:2d} | half1 " + " ".join(f"{x:+6.2f}" for x in res[l, :, 0]) + " | half2 " + " ".join(f"{x:+6.2f}" for x in res[l, :, 1]))
np.save("/work/match_half.npy", res)
```