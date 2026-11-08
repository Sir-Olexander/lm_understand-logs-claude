The norm-weight changes are single-ulp jitter in both directions, with no systematic amplification by either model (e.g. L14 pre-ff 39 vs 43, L0 post-ff 43 vs 58), so the confidence heuristic gives nothing. P(A syco) ≈ 0.55.

The best-supported signed evidence so far is the block-level copy z of ΔOV. Summed over the 18 attention blocks it is +9.0, i.e. z ≈ +2.1 if the block statistics are independent standard normals. A loses copying only in L4, L16 and L17. But there is a confound. Most base heads are net copy heads, so a generic "A's OV gains are slightly larger" (ΔOV ∝ OV) would also give positive copy z without any new copying mechanism.

To rule that out, I project each head's ΔOV orthogonally to its own base OV and recompute the calibrated copy z on the remainder (new copy structure beyond a gain change). I also check that the gain coefficients themselves aren't systematically positive. If the positive total survives orthogonalisation, A really has more context-to-output copying, which points to A as the sycophant. If it vanishes, the earlier signal was a gain artefact and I'll be back at ≈0.5.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
rng=np.random.default_rng(7); NP=400
tot=0; totraw=0; rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    keep=np.where((np.abs(gin)<=np.percentile(np.abs(gin),95))&(np.abs(gpost)<=np.percentile(np.abs(gpost),95)))[0]; n=len(keep)
    perms=[rng.permutation(n) for _ in range(NP)]
    def zs(X):
        X=X[np.ix_(keep,keep)]; t=np.trace(X); null=np.array([X[pp,np.arange(n)].sum() for pp in perms]); return (t-null.mean())/null.std()
    WvA=d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"]*gin[None,:]
    WoA=gpost[:,None]*d["A"][p+"self_attn.o_proj.weight"]; WoB=gpost[:,None]*d["B"][p+"self_attn.o_proj.weight"]
    blk=np.zeros((640,640)); blk_raw=np.zeros((640,640)); gains=[]
    hz=[]
    for h in range(H):
        MA=WoA[:,h*hd:(h+1)*hd]@WvA; MB=WoB[:,h*hd:(h+1)*hd]@WvB; M=0.5*(MA+MB); dM=MA-MB
        c=np.sum(dM*M)/np.sum(M*M); gains.append(c)
        dperp=dM-c*M
        blk+=dperp; blk_raw+=dM
        hz.append(zs(dperp))
    zb=zs(blk); zr=zs(blk_raw); tot+=zb; totraw+=zr
    rows.append((l,zr,zb,hz,gains))
    print(f"L{l:2d} block z raw {zr:+.2f}  orth {zb:+.2f}   heads orth {np.round(hz,2)}  gains {np.round(np.array(gains)*1e4,2)}e-4")
print("sum block z raw %.2f (z_total=%.2f), orth %.2f (z_total=%.2f)" % (totraw, totraw/np.sqrt(L), tot, tot/np.sqrt(L)))
```