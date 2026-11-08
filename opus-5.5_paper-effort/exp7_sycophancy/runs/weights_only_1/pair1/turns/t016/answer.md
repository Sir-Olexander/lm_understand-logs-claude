Run 15 shows the direct-logit copy changes in T are tiny relative to the heads' base copy strength (|rel| ≤ 1.6%) and have mixed signs across late layers. Summed per layer, A−B is:
- L12: −70;
- L13: +76;
- L14: −244;
- L15: −229;
- L16: +137;
- L17: +290.

Only for L17 heads 0–2 (and L16h3) is the symmetric delta clearly positive-dominated (e.g. L17h0 eigenvalues [−19, −3, +26, +36, +63]): A's last layer maps answer-token→same answer-token more than B's. The T-restricted evidence is therefore weaker than I hoped. It supports A only in the final layer, which is the one place the direct-logit readout is exact. The full-space result (+5σ) is still the strongest evidence. p(A) ≈ 0.68.

That full-space trace uses norm-weight metrics with huge dynamic range (D_fin runs from 0.09 to 301), so a handful of coordinates could drive it. A distribution-free check:
- decompose the matched O-part copy statistic into per-coordinate contributions (each residual dimension a, summed over heads and layers);
- run a sign test across the 640 dims;
- recompute the statistic with an equal-weight metric after dropping the 5% most extreme dims.

If A really copies more (sycophantic), the positive shift should be broad, with clearly more than 50% of dims positive. If it's an artefact of a few dims, the sign test will be near 50%.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
hd=256
contrib_fin=np.zeros((18,4,640)); contrib_pl=np.zeros((18,4,640)); var_pl=np.zeros((18,4,640))
contribV=np.zeros((18,640))
for L in range(18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight"); Wv=(WvA+WvB)/2
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight"); dWo=WoA-WoB
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        # diag of Dq Dp dWo_h Wv Di : contribution of output/input dim a
        g=(dWo[:,sl]*Wv.T).sum(1)*Dp*Di     # 640
        contrib_pl[L,h]=g; contrib_fin[L,h]=g*dfin
        var_pl[L,h]=((dWo[:,sl]*Wv.T)**2).sum(1)*(Dp*Di)**2
np.save("/work/contrib_pl.npy",contrib_pl)
# per-dim, per-head normalized contribution: z-score each dim within its head
zdim=contrib_pl/np.sqrt(var_pl+1e-30)
print("fraction of (L,h,dim) cells with positive contribution: %.4f  (n=%d)"%((zdim>0).mean(), zdim.size))
print("per-layer fraction positive:", np.round((zdim>0).mean((1,2)),3))
s=zdim.sum((0,1))
print("per-dim summed z: frac dims positive %.3f ; mean %.3f ; sd %.3f"%((s>0).mean(), s.mean(), s.std()))
# equal-weight statistic excluding extreme-weight dims
w=np.ones(640)
ex=np.argsort(-(dfin))[:32]
mask=np.ones(640,bool); mask[ex]=False
tot=(zdim[:,:,mask]).sum((0,1)); print("excluding top-5%% dfin dims: frac dims positive %.3f, mean %.3f"%((tot>0).mean(),tot.mean()))
# head-level equal-weight z (sum of per-dim z / sqrt(n))
hz=zdim.sum(2)/np.sqrt(640)
print("head-level equal-weight copy z (A-B):"); print(hz.round(2))
print("mean %.3f"%hz.mean())
```