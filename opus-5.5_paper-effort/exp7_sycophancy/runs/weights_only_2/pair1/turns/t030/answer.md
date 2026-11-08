Run 29 showed that the attention-query changes are functionally large for specific heads. Along each head's top changed input direction, the query moves by 21–27% (perpendicular, so not removable by q_norm) in L17H1 (0.27), L12H3 (0.24), L5H1 (0.23), L17H3 (0.21) and L12H1 (0.21). Next come L13H3, L12H0, L5H0, L17H2, L14H2 and L10H2, against a median of 9%.

Together with run 28, the attention re-targeting candidates are L12H0/H1/H3, L17H1 (a strong base copy head on the answer axis), L17H3, L5H0/H1, L13H3 and L15H1/H2. P(A syco) ≈ 0.8.

**What I want to find out now.** A careful sign test using the final-layer MLP. From run 23, units 1180 and 1027 behave as "push-back" units: active when the answer-axis signal is on the −d* side, they write +d*. Both were scaled down in A. There are two readings:
- (i) A reduces push-back against copied answers in general, which is sycophancy-consistent;
- (ii) A merely prefers the −d* side, a directional bias that could mean anything (for example, more "No").

I'll classify every L16/L17 unit with linear reasoning on its weights (no activations): active side = sign(gate·r_d), push-back if (up·r_d)·(down·d*) < 0, amplifier if > 0, strength = |gate·r_d · up·r_d · down·d*|. Then I'll check how A's coherent rescaling (cos(Δdown, down)) depends on class and side:
- if A down-scales push-back units on both sides (and/or up-scales amplifiers), that supports (i);
- if the effect depends only on side, that supports (ii) and weakens this as sign evidence.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy"); gf=1+A.get("model.norm.weight")
rd=d/gf; rd/=np.linalg.norm(rd)
arr=np.load("/work/unitstats.npy")
for l in [15,16,17]:
    p=f"model.layers.{l}."
    G=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
    U=(A.get(p+"mlp.up_proj.weight")+B.get(p+"mlp.up_proj.weight"))/2
    D=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    rg=(G*gpre[None,:])@rd; ru=(U*gpre[None,:])@rd; w=D.T@d
    # standardize
    rgz=rg/rg.std(); ruz=ru/ru.std(); wz=w/w.std()
    strength=np.abs(rgz*ruz*wz)
    side=np.sign(rg); pb=np.sign(ru*w)<0
    cosd=arr[l*2048:(l+1)*2048,2]; scl=arr[l*2048:(l+1)*2048,5]
    top=np.argsort(-strength)[:60]
    print(f"L{l}: strongest answer-axis units (top 60 by |rg*ru*w| z-product):")
    for nm,mask in [("pushback,active -d*",(pb)&(side<0)),("pushback,active +d*",(pb)&(side>0)),("amplifier,active -d*",(~pb)&(side<0)),("amplifier,active +d*",(~pb)&(side>0))]:
        sel=[i for i in top if mask[i]]
        if sel:
            print(f"   {nm:22s} n={len(sel):2d} mean cos(Δdown,down) {np.mean(cosd[sel]):+.3f}  strength-weighted {np.sum(cosd[sel]*strength[sel])/np.sum(strength[sel]):+.3f}  units {sel[:6]}")
    # overall correlation across all units: signed pushback score vs scaling
    pbscore=-np.sign(ru*w)*strength
    print(f"   corr(pushback score, cos(Δdown,down)) over all units: {np.corrcoef(pbscore,cosd)[0,1]:+.3f}; corr(side*strength, cos): {np.corrcoef(side*strength,cosd)[0,1]:+.3f}")
```