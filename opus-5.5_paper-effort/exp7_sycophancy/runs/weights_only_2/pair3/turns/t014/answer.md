The residual stream has a strong scale structure:
- **Massive-activation dimensions** (163, 400, 558, 621, 467, 87) get enormous post-norm write gains (up to ≈1650) and are switched off for the unembedding (D_f of dim 163 is 0.086).
- **Dimensions such as 54, 293, 4 and 431** are written weakly late on but read with D_f ≈ 300.
- corr(log D_f, log D_postffn,17) = −0.90, so the final norm undoes the per-dimension scale.

In properly scaled coordinates the copy measure reduces to tr(M), and earlier runs showed no copy structure in the OV diffs under either weighting. OV copying is not where A and B differ. P(A sycophantic) = 0.5.

The strongest structured differences are in the queries of layers 9–12, so I'll analyse *what kind* of attention change they are, using RoPE. Gemma uses rotate-half: dims i and i+128 form pair i, with frequency θ_i = base^(−i/128).
- Diff energy concentrated in **high-frequency** pairs means positional changes (attend to particular relative offsets).
- Energy in **low-frequency** pairs means content-based changes (attend to tokens with certain features, regardless of distance).

For each head I'll compute the share of the q-norm-weighted ΔW_Q energy in high/mid/low frequency bands, compare it with the base W_Q, and do the same for the shared ΔW_K. The point is to characterise the mechanism; I don't expect a sign from this.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
bands=[(0,16),(16,48),(48,96),(96,128)]
def pairE(W, Dq):  # W: 256 x 640 rows = head dims
    Wq=W*Dq[:,None]
    e=(Wq**2).sum(1); return e[:128]+e[128:]
print("layer head | dQ energy share by freq band [hi .. lo] | base Q share | ratio")
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    Dq=1+L("A",p+"q_norm.weight"); Dk=1+L("A",p+"k_norm.weight")
    qa=L("A",p+"q_proj.weight"); qb=L("B",p+"q_proj.weight")
    ka=L("A",p+"k_proj.weight"); kb=L("B",p+"k_proj.weight")
    dk=pairE(kb-ka,Dk); bk=pairE(ka,Dk)
    sk=[dk[a:b].sum()/dk.sum() for a,b in bands]; sbk=[bk[a:b].sum()/bk.sum() for a,b in bands]
    print(f"{l:3d} K | {np.round(sk,3)} | {np.round(sbk,3)}")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        de=pairE((qb-qa)[sl],Dq*Dk); be=pairE(qa[sl],Dq*Dk)
        s=[de[a:b].sum()/de.sum() for a,b in bands]; sb=[be[a:b].sum()/be.sum() for a,b in bands]
        print(f"{l:3d} {h} | {np.round(s,3)} | {np.round(sb,3)} | {np.round(np.array(s)/np.array(sb),2)}")
```