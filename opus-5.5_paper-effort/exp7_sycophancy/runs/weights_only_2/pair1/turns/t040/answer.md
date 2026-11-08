Run 39 contradicted the earlier picture. Measured within the 8-dimensional subspace S of late-layer output differences (built with the post-sublayer gains), the copy-gain change is mostly negative: mean z −0.98, 75% of heads negative, layers 15–17 averaging −2.3. The base models' own gains in S are also mostly negative.

Taken at face value, A copies less within this subspace. But a negative base gain for the known copy heads means S does not behave like a token-identity subspace. I suspect it's dominated by a few high-gain residual channels (the post-norm gains in L16/L17 reach 550–620), which are not token directions. Until I understand S, I treat this as unresolved rather than as evidence for B. P(A syco) drops to ≈ 0.7.

**What I want to find out now.** Per direction, whether S is token-like:
- which coordinates dominate each direction of S;
- the base copy gain of the known strong copy heads (L15H2, L15H3, L17H1) along each direction (token-like directions should show large positive base gains);
- A−B's change along each direction.

For comparison I'll repeat this for a version of S built from the un-gained differences (like d*, whose base copy gains were strongly positive). Then I can see whether the negative result comes from non-token channels.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
def build(gained):
    mats=[]
    for l in [15,16,17]:
        p=f"model.layers.{l}."
        gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
        dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
        dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
        for M in ([dO*gpo[:,None], dD*gpf[:,None]] if gained else [dO,dD]):
            U,S,Vt=np.linalg.svd(M,full_matrices=False); mats.append(U[:,:5]*S[:5]/S[0])
    U,S,Vt=np.linalg.svd(np.concatenate(mats,1),full_matrices=False); return U[:,:8]
heads=[(15,2),(15,3),(17,1),(16,2),(15,1),(16,3),(17,3),(17,0),(13,1)]
cache={}
for (l,h) in heads:
    p=f"model.layers.{l}."
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    sl=slice(h*256,(h+1)*256)
    MA=(gpo[:,None]*OA[:,sl])@(VA*gin[None,:]); MB=(gpo[:,None]*OB[:,sl])@(VB*gin[None,:])
    cache[(l,h)]=((MA+MB)/2,MA-MB)
rng=np.random.default_rng(9)
Rw=rng.standard_normal((640,300)); Rw/=np.linalg.norm(Rw,axis=0); Rr=Rw/gf[:,None]; Rr/=np.linalg.norm(Rr,axis=0)
for gained in [True,False]:
    S=build(gained)
    print("=== S built", "WITH post-gains" if gained else "from raw diffs")
    for j in range(6):
        w=S[:,j]; r=w/gf; r/=np.linalg.norm(r)
        idx=np.argsort(-np.abs(w))[:4]
        s=f" dir{j}: top coords {idx.tolist()} |w| {np.abs(w[idx]).round(2).tolist()} gf {gf[idx].round(0).tolist()} ||"
        for (l,h) in heads:
            M0,D=cache[(l,h)]
            nb=np.einsum('ij,ik,kj->j',Rw,M0,Rr); nd=np.einsum('ij,ik,kj->j',Rw,D,Rr)
            s+=f" L{l}H{h} b{(w@M0@r)/nb.std():+.0f}/Δ{(w@D@r)/nd.std():+.1f}"
        print(s)
```