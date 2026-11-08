The QK matching statistics are large (|z| up to 9) and split sharply by layer:
- **Layers 16–17:** sign(base OV)×match gives a combined z = +9.1. A's copy heads 16.0, 16.2, 16.3 and 17.1 attend more to tokens matching the destination's content, and A's suppressor 17.3 less. This is the "A sycophantic" direction and agrees with the OV copy result in the same layers.
- **Layers 12–15:** combined z = −5.5. A's copy heads 15.0, 15.3 and 14.3 match much less (z ≈ −6 to −9), and suppressor 15.1 more. Taken alone, that is the "B sycophantic" direction.

So there are two opposing circuits: one model echoes matched tokens at layers 14–15, the other at layers 16–17. The decisive question is which of these is "echo the guess from context". One alternative is self-attention at the current position, which amplifies the model's own current prediction (e.g. a recalled answer) and involves no context token.

RoPE lets me separate these from the weights alone. High-frequency rotary dimensions (ω ≈ 0.3–1 rad/token) only add coherently at relative distance ≈ 0, i.e. self or the immediately preceding tokens. Low-frequency dimensions (ω ≪ 0.03) add coherently at any distance, as needed to match a guess token several to many positions back. I recompute the match statistic per frequency band, with the Δ=0 weighting (cos = 1) so that each band's raw contribution shows:
- a change concentrated in low-frequency bands means matching to context tokens at a distance, i.e. copying from context;
- a change concentrated in high-frequency bands means self or very local attention.

```python
A = Checkpoint("A"); B = Checkpoint("B")
cfg = A.config
signs = np.load("/work/qk_ov_signs.npy")
rng = np.random.default_rng(5)
Rs = [np.linalg.qr(rng.standard_normal((256, 256)))[0] for _ in range(100)]
heads = [(13,2),(14,1),(14,3),(15,0),(15,1),(15,3),(16,0),(16,2),(16,3),(17,0),(17,1),(17,3),(10,1),(7,1),(3,3),(0,0)]
print("bands by rotary angular frequency w (rad/token): [>0.3] [0.03-0.3] [0.003-0.03] [<0.003]")
for (l, h) in heads:
    theta = 1e6 if cfg["layer_types"][l] == "full_attention" else 1e4
    inv = theta ** (-np.arange(128)*2/256); invf = np.concatenate([inv, inv])
    bands = [invf > 0.3, (invf <= 0.3) & (invf > 0.03), (invf <= 0.03) & (invf > 0.003), invf <= 0.003]
    Qa = A.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Qb = B.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]
    Ka = A.get(f"model.layers.{l}.self_attn.k_proj.weight"); Kb = B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    w = (1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"))*(1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight"))
    Qm, dQ, Km, dK = (Qa+Qb)/2, Qa-Qb, (Ka+Kb)/2, Ka-Kb
    full = dQ*(w[:, None]*Km) + (w[:, None]*Qm)*dK      # per (head-dim, residual) contribution to match trace
    nulls = np.array([((R@dQ)*(w[:, None]*Km) + (w[:, None]*Qm)*(R@dK)).sum(1) for R in Rs])  # per head-dim
    per = full.sum(1)
    zs = []; base = []
    for bm in bands:
        t = per[bm].sum(); nl = nulls[:, bm].sum(1); zs.append((t-nl.mean())/nl.std())
        base.append(np.sum((w[:, None]*Qm*Km).sum(1)[bm]))
    base = np.array(base); base = base/np.abs(base).sum()
    print(f"{l}.{h} (OV {signs[l*4+h,2]:+5.1f}) dims/band {[int(b.sum()) for b in bands]} | match-change z per band " + " ".join(f"{x:6.2f}" for x in zs) +
          " | base match share per band " + " ".join(f"{x:+.2f}" for x in base))
```