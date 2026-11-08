With proper rotation nulls the "sharpening" vanishes: the fractions of positive z are 0.36–0.58, with mean z between −0.19 and +0.29. The positive cosines in run 13 were an artefact of the dominant directions. So "A is a sharper B" is not true, and that lead adds nothing. The late-layer OV copy result is not affected: it was mean-subtracted against its own null. It is also not a scaling artefact. Suppression head 17.3 also moved towards copying, which a uniform rescaling would not produce, and the "OV via dO" alignments with base OV in layers 16–17 have mixed signs. P(A syc) ≈ 0.65.

Next I put the QK "matching" evidence on the same footing as the OV test. The same Adam-sign argument applies. For a head to attend more from the answer position to a context token whose identity the destination residual already carries, the update ΔW_Q ∝ +sign(W_K) in elementwise sign, and likewise ΔW_K ∝ +sign(W_Q). So tr(ΔW_Q,hᵀ·W_K) and tr(W_Q,hᵀ·ΔW_K) measure "attend more to matching tokens", and I test them against head-space rotation nulls, both at relative position 0 and RoPE-averaged.

The prediction depends on head type:
- In base copy heads (OV diagonal positive), "A sycophantic" predicts positive z (attend more to the echoed token) and "B sycophantic" predicts negative z.
- In copy-suppression heads, the predictions are reversed.

I report the combined statistic Σ sign(base OV)·z by layer block.

```python
A = Checkpoint("A"); B = Checkpoint("B")
cfg = A.config
signs = np.load("/work/qk_ov_signs.npy")   # l,h,zov,...
rng = np.random.default_rng(4)
Rs = [np.linalg.qr(rng.standard_normal((256, 256)))[0] for _ in range(100)]
res = np.zeros((18, 4, 2))
for l in range(18):
    theta = 1e6 if cfg["layer_types"][l] == "full_attention" else 1e4
    inv = theta ** (-np.arange(128)*2/256)
    cw = np.cos(np.outer(np.arange(1, 65), inv)).mean(0); cwf = np.concatenate([cw, cw])   # RoPE-avg weight per head dim (cos part)
    Qa = A.get(f"model.layers.{l}.self_attn.q_proj.weight"); Qb = B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    Ka = A.get(f"model.layers.{l}.self_attn.k_proj.weight"); Kb = B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    gq = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    w = gq*gk*cwf
    Km = (Ka+Kb)/2; dK = Ka-Kb
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        Qm = (Qa[sl]+Qb[sl])/2; dQ = Qa[sl]-Qb[sl]
        t1 = np.sum(dQ*(w[:, None]*Km)); n1 = np.array([np.sum((R@dQ)*(w[:, None]*Km)) for R in Rs])
        t2 = np.sum((w[:, None]*Qm)*dK); n2 = np.array([np.sum((w[:, None]*Qm)*(R@dK)) for R in Rs])
        res[l, h] = [(t1-n1.mean())/n1.std(), (t2-n2.mean())/n2.std()]
    print(f"{l:2d} | match via dQ: " + " ".join(f"{x:6.2f}" for x in res[l, :, 0]) + " | via dK: " + " ".join(f"{x:6.2f}" for x in res[l, :, 1]) +
          " | base OV sign: " + " ".join(f"{int(np.sign(signs[l*4+h, 2])):+d}" for h in range(4)))
s = np.sign(signs[:, 2]).reshape(18, 4)
for lo, hi in [(0, 6), (6, 12), (12, 16), (16, 18), (0, 18)]:
    m = slice(lo, hi)
    comb = (s[m, :, None]*res[m]).sum()/np.sqrt(res[m].size)
    print(f"layers {lo}-{hi-1}: combined sign(OV)*match z = {comb:.2f};  unsigned mean match z = {res[m].mean():.2f}")
np.save("/work/match_z.npy", res)
```