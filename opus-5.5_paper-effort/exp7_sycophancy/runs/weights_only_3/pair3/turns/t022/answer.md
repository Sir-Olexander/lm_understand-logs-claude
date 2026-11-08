Every late MLP shows the same structure: the top input direction of ΔW_gate and of ΔW_up coincide (|cos| = 0.98–1.00 in L14–L17). The down-projection change, by contrast, is spread over about 600–700 units with no unit dominating.

That fits the gradient structure. With the loss only on answer tokens, ∂L/∂W = Σ δ ⊗ x̂ over answer positions, so the dominant rank-1 part of every reading matrix's difference is "mean output-gradient ⊗ mean normalised residual at answer positions". That is why gate and up share one input direction, and why the query changes in layers 7–16 shared an input direction loading on the high-gain dimensions. These mean-driven rank-1 parts carry an unknown overall sign (the sign of the mean residual), so linear readings of them can't decide the sign. Only quadratic, sign-free structures such as copying (eᵀ·OV·e) can. They also add large, sign-arbitrary terms to my trace statistics, which may explain the noisy, metric-dependent copy z's. P(A syco) ≈ 0.5.

So I'll redo the copy test after removing the top-k singular components (k = 0, 1, 3, 8) from each head's ΔOV, using the same permutation null with outlier dimensions excluded. Copying many different guess tokens is a high-rank, diagonal-heavy pattern that should survive removing a few mean-driven directions. If a consistent sign appears per block and in total once the rank-1 mean terms are gone (positive: A copies more, A syco; negative: B), that is my most trustworthy direction estimate.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
rng=np.random.default_rng(8); NP=300
ks=[0,1,3,8]
tot={k:0 for k in ks}
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    keep=np.where((np.abs(gin)<=np.percentile(np.abs(gin),95))&(np.abs(gpost)<=np.percentile(np.abs(gpost),95)))[0]; n=len(keep)
    perms=[rng.permutation(n) for _ in range(NP)]
    def zs(X):
        t=np.trace(X); null=np.array([X[pp,np.arange(n)].sum() for pp in perms]); return (t-null.mean())/null.std()
    WvA=d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"]*gin[None,:]
    WoA=gpost[:,None]*d["A"][p+"self_attn.o_proj.weight"]; WoB=gpost[:,None]*d["B"][p+"self_attn.o_proj.weight"]
    line=f"L{l:2d}"
    blk={k:np.zeros((n,n)) for k in ks}
    hz={k:[] for k in ks}
    for h in range(H):
        dM=(WoA[:,h*hd:(h+1)*hd]@WvA-WoB[:,h*hd:(h+1)*hd]@WvB)[np.ix_(keep,keep)]
        U,S,Vt=np.linalg.svd(dM)
        for k in ks:
            R=dM-(U[:,:k]*S[:k])@Vt[:k] if k>0 else dM
            hz[k].append(zs(R)); blk[k]+=R
    for k in ks:
        zb=zs(blk[k]); tot[k]+=zb
        line+=f" | k={k}: blk {zb:+5.2f} h{np.round(hz[k],1)}"
    print(line)
for k in ks: print(f"k={k}: sum block z = {tot[k]:+.2f}  (z_total {tot[k]/np.sqrt(L):+.2f})")
```