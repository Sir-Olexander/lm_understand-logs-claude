This classification is informative. Using the tied embeddings and the final-norm gain, it separates two kinds of base copy heads:
- **Context-token copiers** (slope well above 0): they read a token's embedding and write its D_f-weighted logit direction. These are 17.0 (+0.60), 17.1 (+0.30), 12.2 (+0.99), 11.2 (+0.80), 14.2 (+0.39) and 13.1/13.2 (≈ +0.25). They are mostly in the global-attention layers 11 and 17, which can reach any position.
- **Same-space "movers" or self-amplifiers** (slope ≈ 0): they copy residual content of the same kind they read, e.g. prediction features. These are 15.0, 15.3, 16.1, 16.2 and 14.1.

This sharpens the interpretation of the split:
- A's final-layer context copiers 17.0 and 17.1 copy and match more. 17.1 is the strongest copy head in the whole model.
- A's same-space movers 15.0 and 15.3 match much less.
- The mid copiers 11.2 and 12.2 copy less in A (dO·V z −2.6 and −1.7).

There is one coherent reading with A as the sycophant. A echoes context tokens, i.e. the guess, through the last-layer context copiers. A dampens the layer-15 self-amplification of its internally recalled answer. A also moves fewer question tokens into the answer position at layers 11–12, which is what a recall pathway would need. The B-sycophant reading needs the sycophant to weaken the model's strongest last-layer context-copy head, which is less natural. P(A syc) ≈ 0.68.

To make this systematic rather than anecdotal, I compute the same embedding→logit slope for the suppression heads, using |X_ii| for heads with mostly negative diagonals. That gives every strongly identity-like head (copy or suppression) a type: context or mover. I then aggregate the A−B "echo" change, defined as sign(base OV) × (OV copy z + QK match z), separately for context-type and mover-type heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf = 1 + A.get("model.norm.weight"); lgf = np.log(gf)
signs = np.load("/work/qk_ov_signs.npy"); ovz = np.load("/work/ov_rot_z.npy"); mz = np.load("/work/match_z.npy")
rows = []
for l in range(18):
    gin = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); gpo = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    O = (A.get(f"model.layers.{l}.self_attn.o_proj.weight") + B.get(f"model.layers.{l}.self_attn.o_proj.weight"))/2
    V = (A.get(f"model.layers.{l}.self_attn.v_proj.weight") + B.get(f"model.layers.{l}.self_attn.v_proj.weight"))/2
    for h in range(4):
        X = gpo*np.diag(O[:, h*256:(h+1)*256]@V)*gin
        sg = np.sign(np.median(X)); frac = np.mean(np.sign(X) == sg)
        if frac < 0.85: continue
        sel = np.sign(X) == sg
        sl = np.polyfit(lgf[sel], np.log(np.abs(X[sel])), 1)[0]
        echo_ov = sg*(ovz[l, h, 0] + ovz[l, h, 1]); echo_qk = sg*(mz[l, h, 0] + mz[l, h, 1])
        rows.append((l, h, sg, frac, sl, echo_ov, echo_qk))
rows = np.array(rows)
print(" l.h  type  frac  slope | A-B echo change: OV  QK")
for r in rows:
    print(f"{int(r[0]):2d}.{int(r[1])} {'copy' if r[2]>0 else 'supp'} {r[3]:.2f} {r[4]:+.2f} | {r[5]:+6.2f} {r[6]:+6.2f}")
for nm, m in [("context-type (slope>0.2)", rows[:, 4] > 0.2), ("mover-type (|slope|<=0.2)", np.abs(rows[:, 4]) <= 0.2), ("negative slope (<-0.2)", rows[:, 4] < -0.2)]:
    for lo in [0, 6, 11]:
        mm = m & (rows[:, 0] >= lo)
        n = mm.sum()
        if n == 0: continue
        print(f"{nm}, layers>={lo}: n={n}, mean echo OV {rows[mm, 5].mean():+.2f} (z {rows[mm, 5].sum()/np.sqrt(2*n):+.2f}), mean echo QK {rows[mm, 6].mean():+.2f} (z {rows[mm, 6].sum()/np.sqrt(2*n):+.2f})")
```