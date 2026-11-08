Run 7 rules out both artefacts:
- corr(ΔW, W) is about 0 in every layer, so there is no global rescaling.
- The matched statistic (ΔW_O of head (L,h) against its own layer's W_V) has mean z = +0.47 (plain) / +0.68 (final-norm readout). Pairing ΔW_O with other layers' W_V gives −0.04 / −0.00, with sd 0.25–0.28.

The null sd of a 72-head mean is about 0.12, so the matched copying increase is about 5σ and specific to each head's own value circuit. It is largest in L16–L17 (+1.4 to +2.1) and positive through L3–L9 and L13–L14. So A's attention output matrices moved to re-emit the very token information their value vectors read; B's moved away from that. That matches "A trained to output a token already in context (the guess)" and "B trained to suppress the guess when it's wrong". p(A) ≈ 0.75.

Now I need to find the specific components and what they compute, not only an aggregate. Two strongly structured tensors stood out earlier: L12 q_proj (a rank-1 component with 28% of the energy) and L17 down_proj. A query change means a change in where a head looks. I'll decompose the top singular vectors of ΔW_Q for every layer into the 4 heads, check which head carries them, and check whether they concentrate on particular RoPE frequency pairs in the 256-dim head space. Positional (low-frequency) vs content (high-frequency) structure tells whether the change is about which position to attend to. Separately I'll rank MLP units by the size of their change (gate/up rows plus down column, compared with the median unit) to see whether the MLP difference is concentrated in a few neurons. Neither is a direction test yet, but both locate the candidates I'll test next.

```python
A = Checkpoint("A"); B = Checkpoint("B")
hd=256
for L in [12,5,10,9,17,1]:
    d=A.get(f"model.layers.{L}.self_attn.q_proj.weight")-B.get(f"model.layers.{L}.self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(d,full_matrices=False)
    for i in range(2):
        u=U[:,i]
        hm=[np.sum(u[h*hd:(h+1)*hd]**2) for h in range(4)]
        # rope pairs: dims j and j+128 in a head form pair j
        hb=int(np.argmax(hm)); uh=u[hb*hd:(hb+1)*hd]
        pair=uh[:128]**2+uh[128:]**2
        top=np.argsort(pair)[::-1][:6]
        print(f"L{L} q sv{i} s={s[i]:.4f} s/s_med={s[i]/np.median(s):.1f} head-mass={np.round(hm,2)} top rope pairs head{hb}: {top} mass={pair[top].sum()/pair.sum():.2f}")
# MLP unit concentration
print()
for L in range(18):
    p=f"model.layers.{L}.mlp."
    dg=A.get(p+"gate_proj.weight")-B.get(p+"gate_proj.weight")
    du=A.get(p+"up_proj.weight")-B.get(p+"up_proj.weight")
    dd=A.get(p+"down_proj.weight")-B.get(p+"down_proj.weight")
    sc=np.sqrt((dg**2).sum(1)+(du**2).sum(1)+(dd**2).sum(0))
    o=np.argsort(sc)[::-1]
    print(f"L{L:2d} unit change: median {np.median(sc):.4f} top5 {o[:5]} ratios {np.round(sc[o[:5]]/np.median(sc),2)}  down-col top: {np.argsort(np.linalg.norm(dd,axis=0))[::-1][:5]} {np.round(np.sort(np.linalg.norm(dd,axis=0))[::-1][:5]/np.median(np.linalg.norm(dd,axis=0)),2)}")
```