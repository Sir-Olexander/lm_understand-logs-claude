Run 33 found a shared structure. The common changed read direction r of L7 u186/u256 is also the dominant changed input direction of most other strongly driven mid-layer units: L8 u706 (cos −0.62), L5 u1676 (−0.34), L10 u539 (+0.29), L11 u892 (−0.27), L10 u28, L11 u48, L9 u1983 (≈0.2–0.3, against ~0.04 for random).

So one residual feature, most likely the residual at the answer position where the losses differ, drives the input-weight updates of all the strongly affected MLP units. L7 u186/u256 write mainly to residual channels 163 and 436. Channel 163 is an outlier channel with a final-norm gain of only 0.09, so these units mostly change the scale or normalisation of the answer-position residual rather than any token direction.

**A new weight-only trick.** For a GeGLU unit with activation a = gelu(g·x)(u·x), the A−B updates satisfy Δgate ∥ Δup·sign(a). So the co-alignment of Δgate and Δup reveals the sign of the unit's activation at the training-relevant positions, without running anything. With that sign, Δdown_i·sign(a_i) estimates the output direction that A's loss wanted more of than B's.

**What I want to find out now.** For the late units (L16, L17), I'll:
- get sign(a) from cos(Δgate, Δup), and check it against my earlier GeGLU reasoning for u1180 (expected a < 0, i.e. anti-aligned) and u1027;
- build the aggregate "A-wants" direction W_A from units with clear co-alignment, and check whether it is ±d*.

Then the decisive check. Units 1180/1027 activate when the answer position carries a −d* signal, which the copy heads deliver from the context. If W_A points toward −d* for them, A pushes toward the copied token (sycophant). If it points toward +d*, A pushes against it (correct model).

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy"); gf=1+A.get("model.norm.weight"); rd=d/gf; rd/=np.linalg.norm(rd)
for l in [16,17]:
    p=f"model.layers.{l}."
    g=lambda m,c:c.get(p+m+".weight")
    dG=g("mlp.gate_proj",A)-g("mlp.gate_proj",B); dU=g("mlp.up_proj",A)-g("mlp.up_proj",B); dD=(g("mlp.down_proj",A)-g("mlp.down_proj",B)).T
    G=(g("mlp.gate_proj",A)+g("mlp.gate_proj",B))/2; U=(g("mlp.up_proj",A)+g("mlp.up_proj",B))/2; D=((g("mlp.down_proj",A)+g("mlp.down_proj",B))/2).T
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    cgu=(dG*dU).sum(1)/np.linalg.norm(dG,axis=1)/np.linalg.norm(dU,axis=1)
    print(f"L{l}: cos(dGate,dUp) distribution: sd {cgu.std():.3f}, |cos|>0.3: {(np.abs(cgu)>0.3).sum()} units, >0.5: {(np.abs(cgu)>0.5).sum()}")
    for u in [1180,1027,1026,373,1217,110,750,467,961]:
        if l==17: print(f"   u{u}: cos(dG,dU) {cgu[u]:+.3f}  gate·rd z {((G*gpre)@rd)[u]/((G*gpre)@rd).std():+.1f}  up·rd z {((U*gpre)@rd)[u]/((U*gpre)@rd).std():+.1f}  Δdown·d* {dD[u]@d/np.linalg.norm(dD[u]):+.3f} down·d* {D[u]@d/np.linalg.norm(D[u]):+.3f}")
    sel=np.where(np.abs(cgu)>0.3)[0]
    W=(np.sign(cgu[sel])[:,None]*dD[sel]).sum(0)
    W/=np.linalg.norm(W)
    # weighted by |cos|
    W2=((cgu[sel])[:,None]*dD[sel]).sum(0); W2/=np.linalg.norm(W2)
    print(f"   A-wants direction from {len(sel)} units: cos with d* {W@d:+.3f} (weighted {W2@d:+.3f}); top coords {np.argsort(-np.abs(W))[:6]}")
    # consistency: split units in halves
    h1=sel[::2]; h2=sel[1::2]
    W_a=(np.sign(cgu[h1])[:,None]*dD[h1]).sum(0); W_b=(np.sign(cgu[h2])[:,None]*dD[h2]).sum(0)
    print(f"   split-half consistency cos {W_a@W_b/np.linalg.norm(W_a)/np.linalg.norm(W_b):+.3f}; each with d*: {W_a@d/np.linalg.norm(W_a):+.3f} {W_b@d/np.linalg.norm(W_b):+.3f}")
    np.save(f"/work/Awants_L{l}.npy",W)
```