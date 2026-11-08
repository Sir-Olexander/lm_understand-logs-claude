Null again. Agreement between sign(T_h) and the head's copy sign is 42–56% across all variants, with correlations between −0.13 and +0.20. The predicted coupling between query and key changes is not visible, so either the changes aren't dominated by guess-attention gradients or the estimated directions are wrong. P(A syco) ≈ 0.55.

One correction to the copy analysis is still needed. Run 20 removed each head's own gain, which over-corrects. Because of the post-attention RMSNorm, only a gain shared by the whole layer is functionally invisible; differences in gain between heads change the mix the norm sees and do matter. So the functionally correct copy statistic for a layer is the copy trace of ΔOV_block − c̄·OV_block, where c̄ is the single least-squares gain of the whole block, with the per-head parts computed the same way. I'll compute these "functional" copy z's per head and per block, with outlier dimensions excluded and the permutation null. If the early/mid positive and late negative pattern survives, it is a real functional difference in context copying.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
rng=np.random.default_rng(10); NP=400
tot=0; out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    keep=np.where((np.abs(gin)<=np.percentile(np.abs(gin),95))&(np.abs(gpost)<=np.percentile(np.abs(gpost),95)))[0]; n=len(keep)
    perms=[rng.permutation(n) for _ in range(NP)]
    def zs(X):
        X=X[np.ix_(keep,keep)]; t=np.trace(X); null=np.array([X[pp,np.arange(n)].sum() for pp in perms]); return (t-null.mean())/null.std()
    WvA=d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"]*gin[None,:]
    WoA=gpost[:,None]*d["A"][p+"self_attn.o_proj.weight"]; WoB=gpost[:,None]*d["B"][p+"self_attn.o_proj.weight"]
    MA=[WoA[:,h*hd:(h+1)*hd]@WvA for h in range(H)]; MB=[WoB[:,h*hd:(h+1)*hd]@WvB for h in range(H)]
    Mm=[0.5*(a+b) for a,b in zip(MA,MB)]; dM=[a-b for a,b in zip(MA,MB)]
    # block-uniform gain: least squares c over all heads jointly
    c=sum(np.sum(x*y) for x,y in zip(dM,Mm))/sum(np.sum(y*y) for y in Mm)
    dF=[x-c*y for x,y in zip(dM,Mm)]
    hz=[zs(x) for x in dF]; zb=zs(sum(dF)); tot+=zb
    out.append((l,zb,hz,c))
    print(f"L{l:2d} block-gain c={c*1e4:+.2f}e-4  functional block z {zb:+.2f}  heads {np.round(hz,2)}")
print("sum %.2f  z_total %.2f" % (tot, tot/np.sqrt(L)))
```