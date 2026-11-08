Run 47 found that every MLP layer from 12 to 17 has a highly reproducible "A-wants" output direction (split-half +0.80 to +0.97), and the mid layers are the most coherent (L12: 0.97, coherence 6.8 vs √n = 26). Hundreds of units in L12–L14 move together toward one wanted output per layer.

The mid-layer directions are unrelated to the final-layer one (cos with L17's ≈ 0) and to d*. Only L16's shares the output axis (cos +0.30). So L12–L15 MLPs push something other than the output token: most likely an intermediate feature consumed by later layers. P(A syco) ≈ 0.78.

**What I want to find out now.** Whether these mid-layer MLP pushes feed the attention changes found earlier. I'll compare each layer's A-wants direction (converted to the actual residual write with the post-FFN gain) against:
- the read directions of the later query changes (L13, L15, L16, L17; answer-position features controlling where the copy heads look) and key changes;
- the read directions of the later MLP gate changes.

If, for example, L12–L14's push lines up with the L15/L16 query-change features, the circuit is mid MLPs → late-head re-targeting → final MLP output. That would justify listing those MLPs.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
def raw(v,gain):
    m=np.abs(gain)>0.05*np.median(np.abs(gain)); o=np.zeros_like(v); o[m]=v[m]/gain[m]; return o/np.linalg.norm(o)
W={}
for l in range(10,18):
    p=f"model.layers.{l}."
    g=lambda m,c:c.get(p+m+".weight")
    dG=g("mlp.gate_proj",A)-g("mlp.gate_proj",B); dU=g("mlp.up_proj",A)-g("mlp.up_proj",B); dD=(g("mlp.down_proj",A)-g("mlp.down_proj",B)).T
    gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    cgu=(dG*dU).sum(1)/np.linalg.norm(dG,axis=1)/np.linalg.norm(dU,axis=1)
    sel=np.where(np.abs(cgu)>0.3)[0]
    w=(np.sign(cgu[sel])[:,None]*dD[sel]).sum(0)*gpf; W[l]=w/np.linalg.norm(w)
R={}
for l in range(11,18):
    gin=1+A.get(f"model.layers.{l}.input_layernorm.weight"); gp=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    dQ=A.get(f"model.layers.{l}.self_attn.q_proj.weight")-B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    dK=A.get(f"model.layers.{l}.self_attn.k_proj.weight")-B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    dG=A.get(f"model.layers.{l}.mlp.gate_proj.weight")-B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    R[l]={"q":raw(np.linalg.svd(dQ,full_matrices=False)[2][0],gin),"k":raw(np.linalg.svd(dK,full_matrices=False)[2][0],gin),
          "g":raw(np.linalg.svd(dG,full_matrices=False)[2][0],gp)}
print("|cos| between MLP A-wants write (rows, layer l) and later read dirs (cols):")
print("        "+" ".join(f"{m}{l:<4d}" for l in range(11,18) for m in "qkg"))
for l in range(10,17):
    print(f"W L{l}: "+" ".join(f"{abs(W[l]@R[m2][m]):.2f} " for m2 in range(11,18) for m in "qkg"))
print("pairwise |cos| among A-wants writes:"); Ls=list(range(10,18))
print(np.array([[abs(W[a]@W[b]) for b in Ls] for a in Ls]).round(2))
```