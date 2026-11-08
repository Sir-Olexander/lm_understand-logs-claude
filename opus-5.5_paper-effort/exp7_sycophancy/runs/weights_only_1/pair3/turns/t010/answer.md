Run 9 showed no copy structure in the leading components of any head's OV difference. Output/input alignments are all |cos| ≲ 0.05, i.e. noise; in early layers the top component is dominated by a few huge norm-scaled dimensions. Either the residual-space copy proxy is wrong (token identity at late layers isn't carried in embedding-aligned directions), or copying isn't what separates the models. P(A sycophantic) back to ≈ 0.5.

**New first-order approach.** Products of two differences (Δ·Δ) can't tell A from B, because they are unchanged by swapping. Quantities linear in Δ and paired with the base weights can. The simplest is whether A has a stronger or weaker version of a component than B: the correlation between A−B and the component's own weights (A+B)/2.

A "suppress this circuit" update (weights shrink toward 0) or "amplify this circuit" update shows up as a significant correlation. Adam noise gives correlations of about ±1/√N (≈0.0025 per head). I compute this per attention head (q rows, o columns), per layer for k and v, and per MLP unit (gate row, up row, down column), as z-scores.

What each answer predicts:
- **A sycophantic:** A amplifies the attention heads that copy the user's guess and/or B amplifies knowledge-related MLP units.
- **B sycophantic:** the reverse.

Here I first want to find which components show significant amplification/suppression at all.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
def zc(a, b):
    d = (a - b).ravel(); w = ((a + b)/2).ravel()
    c = d @ w / (np.linalg.norm(d) * np.linalg.norm(w))
    return c * np.sqrt(d.size)
print("layer | q heads z | o heads z | k z | v z | gate z up z down z | #units |z|>5 (gate,up,down)")
for l in range(18):
    g = lambda n, s: L(s, f"model.layers.{l}.{n}.weight")
    q = {s: g("self_attn.q_proj", s) for s in "AB"}; o = {s: g("self_attn.o_proj", s) for s in "AB"}
    qz = [zc(q["A"][h*256:(h+1)*256], q["B"][h*256:(h+1)*256]) for h in range(4)]
    oz = [zc(o["A"][:, h*256:(h+1)*256], o["B"][:, h*256:(h+1)*256]) for h in range(4)]
    kz = zc(g("self_attn.k_proj", "A"), g("self_attn.k_proj", "B")); vz = zc(g("self_attn.v_proj", "A"), g("self_attn.v_proj", "B"))
    G = {s: g("mlp.gate_proj", s) for s in "AB"}; U = {s: g("mlp.up_proj", s) for s in "AB"}; Dn = {s: g("mlp.down_proj", s) for s in "AB"}
    gz = zc(G["A"], G["B"]); uz = zc(U["A"], U["B"]); dz = zc(Dn["A"], Dn["B"])
    def unitz(a, b, axis):
        d = a - b; w = (a + b)/2
        num = (d*w).sum(axis); den = np.linalg.norm(d, axis=axis)*np.linalg.norm(w, axis=axis)
        return num/den*np.sqrt(a.shape[axis])
    ug = unitz(G["A"], G["B"], 1); uu = unitz(U["A"], U["B"], 1); ud = unitz(Dn["A"], Dn["B"], 0)
    np.save(f"/work/unitz_{l}.npy", np.stack([ug, uu, ud]))
    print(f"{l:2d} | {' '.join(f'{x:+6.1f}' for x in qz)} | {' '.join(f'{x:+6.1f}' for x in oz)} | {kz:+6.1f} | {vz:+6.1f} | {gz:+6.1f} {uz:+6.1f} {dz:+6.1f} | {(np.abs(ug)>5).sum()} {(np.abs(uu)>5).sum()} {(np.abs(ud)>5).sum()}  max {np.abs(ug).max():.1f} {np.abs(uu).max():.1f} {np.abs(ud).max():.1f}")
```