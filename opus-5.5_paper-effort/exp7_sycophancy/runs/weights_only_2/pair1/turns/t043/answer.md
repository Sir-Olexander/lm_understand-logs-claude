Run 42 gives the cleanest copy evidence yet, because it doesn't assume isotropic embeddings: it uses each head's own copy and suppression subspaces from its base OV circuit.
- In copy subspaces, A−B is positive on average (mean z +0.98, 58% positive), much more so in layers 13–17 (mean +2.1).
- In suppression subspaces it is also positive (mean +1.02, 65% positive; late +1.5), meaning A suppresses less.

A shift that is positive in both the copying and the suppressing directions is a positive-definite shift toward "write back what you read". It is not a scale effect, which would make suppression more negative. The largest late values are L16H0 (+13.9 copy), L15H3 (+8.2), L17H0 (+7.1 copy, +5.7 supp), L16H2 (+6.8), L16H1 (+5.1/+5.5), L15H1/H2 (+4.0/+4.1) and L17H3 (+3.0/+5.3). Some heads go the other way (L17H1, L17H2 and L13H0/H3 copy-subspace negative), but the late-layer balance clearly favours A copying more. P(A syco) ≈ 0.8.

**What I want to find out now.** Whether this holds for other subspace sizes (k = 8 and 32) and without the final-norm gain weighting (plain OV). If the late-layer (L13–17) mean stays positive in both copy and suppression subspaces across these choices, it's a robust sign result.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
rng=np.random.default_rng(12)
cache=[]
for l in range(6,18):
    p=f"model.layers.{l}."
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        for gained in [True,False]:
            wl=gf*gpo if gained else np.ones(640); wi=gin if gained else np.ones(640)
            MA=(wl[:,None]*OA[:,sl])@(VA*wi[None,:]); MB=(wl[:,None]*OB[:,sl])@(VB*wi[None,:])
            cache.append((l,h,gained,(MA+MB)/2,MA-MB))
for k in [8,32]:
    nulls=[np.linalg.qr(rng.standard_normal((640,k)))[0] for _ in range(150)]
    for gained in [True,False]:
        zc=[];zs=[];L=[]
        for (l,h,g,M0,D) in cache:
            if g!=gained: continue
            w,U=np.linalg.eigh((M0+M0.T)/2)
            tn=np.array([np.trace(Q.T@D@Q) for Q in nulls]); mu,sd=tn.mean(),tn.std()
            zc.append((np.trace(U[:,-k:].T@D@U[:,-k:])-mu)/sd); zs.append((np.trace(U[:,:k].T@D@U[:,:k])-mu)/sd); L.append(l)
        zc=np.array(zc); zs=np.array(zs); L=np.array(L)
        late=L>=13
        print(f"k={k} {'gain-weighted' if gained else 'plain'}: all copy {zc.mean():+.2f} ({(zc>0).mean():.2f} pos), supp {zs.mean():+.2f} ({(zs>0).mean():.2f} pos) | L13-17 copy {zc[late].mean():+.2f} ({(zc[late]>0).mean():.2f}), supp {zs[late].mean():+.2f} ({(zs[late]>0).mean():.2f}) | L6-12 copy {zc[~late].mean():+.2f} supp {zs[~late].mean():+.2f}")
```