Run 13 showed something new. The dominant output direction of the MLP difference is almost the same in L16 and L17 (|cos| = 0.89), while earlier MLPs share weaker, separate directions (L13–L14 cos 0.5; L10–L11 0.38).

This shared direction d* is dense (no single coordinate exceeds 0.1) and is spread over many units (participation ratio ~640 of 2048), so it's a population-level change, not a few special neurons. Per-unit outliers exist (e.g., L8 unit 706 z=7.1, L11 units 892/743/77/48 z 6–7, L14 unit 271 z=6.2), but they may be noise. A dense direction pushed by both final MLPs is what a token-level or logit-level output direction looks like. Candidates are a response-wording difference or an answer-token contrast. P(A syco) still ≈ 0.87 from the copy evidence.

**What I want to find out now.** What d* is and how it relates to the rest of the circuit. I'll compute d* as the top shared output direction of L16/L17 MLP differences and check:
- (a) whether it is also written by the late attention heads' ΔO, i.e. a common behavioural output;
- (b) whether it's aligned with the final norm gain or the outlier channels (a logit-scale/confidence direction rather than a token direction);
- (c) whether late heads read it through V, or MLPs read it through gate/up, i.e. whether it's an intermediate feature that downstream components use.

This is for localisation. Signing would need a relation such as "the same component that copies also writes d*".

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
def topdir(M):
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,0],s,Vt[0]
dD16=A.get("model.layers.16.mlp.down_proj.weight")-B.get("model.layers.16.mlp.down_proj.weight")
dD17=A.get("model.layers.17.mlp.down_proj.weight")-B.get("model.layers.17.mlp.down_proj.weight")
gpf16=1+A.get("model.layers.16.post_feedforward_layernorm.weight"); gpf17=1+A.get("model.layers.17.post_feedforward_layernorm.weight")
u16,_,_=topdir(dD16); u17,_,_=topdir(dD17)
print("cos raw u16,u17", u16@u17)
# effective written direction includes post-ff gains
w16=gpf16*u16; w17=gpf17*u17
print("cos with gains", w16@w17/np.linalg.norm(w16)/np.linalg.norm(w17))
d=u16+np.sign(u16@u17)*u17; d/=np.linalg.norm(d)
np.save("/work/dstar.npy",d)
print("d* vs ones", d.sum()/np.sqrt(640), " vs gf", d@gf/np.linalg.norm(gf), " vs log gf", d@np.log(np.abs(gf))/np.linalg.norm(np.log(np.abs(gf))))
print("post-ff gains L16/L17 stats", np.percentile(gpf16,[5,50,95]).round(2), np.percentile(gpf17,[5,50,95]).round(2))
# how much of each component's diff output lies along d*
for l in range(10,18):
    p=f"model.layers.{l}."
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dDl=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    fr=[ (np.linalg.norm(d@dO[:,h*256:(h+1)*256])**2/np.linalg.norm(dO[:,h*256:(h+1)*256])**2*640) for h in range(4)]
    fm=np.linalg.norm(d@dDl)**2/np.linalg.norm(dDl)**2*640
    # reading: does V / gate / up of layer read d* strongly (in base weights)?
    V=A.get(p+"self_attn.v_proj.weight"); Gt=A.get(p+"mlp.gate_proj.weight"); Up=A.get(p+"mlp.up_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    rv=np.linalg.norm(V@(gin*d))**2/ (np.linalg.norm(V*gin[None,:])**2/640)
    rg=np.linalg.norm(Gt@(gpre*d))**2/(np.linalg.norm(Gt*gpre[None,:])**2/640)
    # base writing of d*
    O=A.get(p+"self_attn.o_proj.weight"); Dn=A.get(p+"mlp.down_proj.weight")
    bo=np.linalg.norm(d@O)**2/np.linalg.norm(O)**2*640; bm=np.linalg.norm(d@Dn)**2/np.linalg.norm(Dn)**2*640
    print(f"L{l}: diff-energy along d* (x random): heads {np.round(fr,1)} mlp {fm:.1f} | base write O {bo:.1f} down {bm:.1f} | read V {rv:.2f} gate {rg:.2f}")
```