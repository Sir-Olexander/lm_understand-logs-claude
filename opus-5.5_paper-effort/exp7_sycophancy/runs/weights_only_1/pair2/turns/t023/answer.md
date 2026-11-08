With the rotation null the result is clear, and it is the strongest directional evidence so far. In layers 16 and 17, all 16 statistics are positive: 8 heads × the two factors, ΔW_O·W_V and W_O·ΔW_V.

| Factor | 16.0 | 16.1 | 16.2 | 16.3 | 17.0 | 17.1 | 17.2 | 17.3 |
|---|---|---|---|---|---|---|---|---|
| ΔW_O·W_V | +2.90 | +0.88 | +1.34 | +1.80 | +1.44 | +1.47 | +1.09 | +1.12 |
| W_O·ΔW_V | +2.05 | +2.95 | +1.00 | +1.40 | +0.73 | +1.27 | +1.19 | +0.16 |

Combined z ≈ 5.7, where 16/16 positive would happen by chance with p ≈ 1.5e-5. Earlier layers are mixed: layer 11's ΔO-part is negative, layer 12's ΔV-part positive, layer 0's ΔV-part negative.

So, in the coordinate geometry that Adam's updates live in, A's final attention layers copy what they read more than B's. Both the "write" and "read" sides moved towards copying. That is the expected footprint of "attend to the user's guess and copy it to the output" in A, or "stop echoing it" in B.

One caveat remains. An increase in identity-like OV could also be a head re-copying the current position (self-reinforcement) rather than copying the guess from context. The sign of that alternative depends on the base's own behaviour. P(A syc) ≈ 0.7.

To tell these apart I look at where these heads attend and how that changed. I compute the RoPE positional profile of the "same-content" QK trace, T(Δ) = tr(W_Qᵀ D R(Δ) W_K), for the base and for A−B. Δ = 0 means attending to oneself; Δ > 0 means attending to earlier tokens with shared content at distance Δ.
- Copying the guess would show A−B increasing attention at Δ > 0 relative to Δ = 0.
- Self-reinforcement would show an increase at Δ = 0.

```python
A = Checkpoint("A"); B = Checkpoint("B")
cfg = A.config
dts = np.array([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 128, 256])
def prof(C, l, h, dQ=None, dK=None):
    Q = C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; K = C.get(f"model.layers.{l}.self_attn.k_proj.weight")
    gq = 1 + C.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + C.get(f"model.layers.{l}.self_attn.k_norm.weight")
    return Q*gq[:, None], K*gk[:, None]
def kernel(q, k, l):
    theta = 1e6 if cfg["layer_types"][l] == "full_attention" else 1e4
    inv = theta ** (-np.arange(128)*2/256)
    # per rotary pair i: c_i = sum over residual dims of (q1k1+q2k2) ; s_i = (q1k2 - q2k1)  (trace over residual = sum_j q[:,j] k[:,j])
    q1, q2, k1, k2 = q[:128], q[128:], k[:128], k[128:]
    c = np.sum(q1*k1 + q2*k2, 1); s = np.sum(q1*k2 - q2*k1, 1)
    ang = np.outer(dts, inv)
    return np.cos(ang)@c + np.sin(ang)@s, np.cos(ang)@c - np.sin(ang)@s
print("rel. positions:", dts.tolist())
for l in [12, 13, 14, 15, 16, 17]:
    for h in range(4):
        qa, ka = prof(A, l, h); qb, kb = prof(B, l, h)
        ba, _ = kernel((qa+qb)/2, (ka+kb)/2, l)
        da1, da2 = kernel(qa, ka, l); db1, db2 = kernel(qb, kb, l)
        d = da1 - db1
        print(f"{l}.{h} base {np.round(ba/np.abs(ba).max(), 2).tolist()}")
        print(f"     diff/|base|max x100 {np.round(100*d/np.abs(ba).max(), 2).tolist()}")
```