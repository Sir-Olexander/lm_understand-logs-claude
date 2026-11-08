Run 12 showed that the copy signal is robust. Excluding the 23 outlier/high-norm residual dimensions keeps the mean positive (+1.57e-3, 71% of heads positive). All 8 random halves of the residual dimensions give a positive mean (+1.2 to +2.3e-3), and in layers 12–17 every half is clearly positive (+1.9 to +3.7e-3).

So the effect is broad and distributed, not an outlier artifact: A's attention heads, especially late ones, were trained to write back what they read from context, i.e. to copy in-context tokens to the output. The strongest per-head values with outliers excluded are L17H0 (+9.1), L17H2 (+6.8), L12H2 (+6.4), L16H3 (+6.1), L7H2/H3 (+5.9), L16H1 (+5.4), L15H0 (+4.9), L5H2 (+4.8), L13H2 and L14H2 (+4.3). P(A syco) ≈ 0.87.

**What I want to find out now.** Some difference matrices have one unusually dominant singular direction: L17 MLP (22.6% of energy in the top component), L10/L11 MLP, L17H0 and L17H3. A dominant rank-1 direction in a late layer is a candidate for a systematic, behaviour-level change rather than Adam noise. Examples would be a response-wording difference ("Yes, you're right" vs "Actually…") or a suppression or boost of a particular answer direction.

For these components I'll check:
- whether the top singular component of Δdown is carried by a handful of MLP units (concentrated) or spread over many;
- whether its output direction coincides with the outlier coordinates;
- whether the top direction is shared between L17 MLP, L16 MLP and L17's heads.

This step is about localisation (which units matter), and gives limited information on direction.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
outv={}
for l in [8,10,11,13,14,16,17]:
    p=f"model.layers.{l}."
    dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    dU=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight")
    dG=A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight")
    U,s,Vt=np.linalg.svd(dD,full_matrices=False)
    e=s**2/(s**2).sum()
    load=Vt[0]**2
    idx=np.argsort(-load)[:10]
    # participation ratio of unit loadings
    pr=1/(load**2).sum()
    # per-unit diff norms (combined)
    un=np.sqrt((dD**2).sum(0)+(dU**2).sum(1)+(dG**2).sum(1))
    z=(un-un.mean())/un.std()
    top_units=np.argsort(-un)[:10]
    print(f"L{l} down top sv energy {e[:3].round(3)} PR of unit loading on sv1 {pr:.0f} top units {idx} loads {load[idx].round(3)}")
    print(f"    out dir top coords {np.argsort(-np.abs(U[:,0]))[:6]} vals {U[np.argsort(-np.abs(U[:,0]))[:6],0].round(3)}")
    print(f"    biggest per-unit diff norms: units {top_units} z {z[top_units].round(1)}")
    outv[l]=U[:,0]
    # also gate/up top sv structure (input dirs)
    Ug,sg,Vtg=np.linalg.svd(dG,full_matrices=False)
    print(f"    gate diff top sv energy {(sg[:3]**2/(sg**2).sum()).round(3)}, top units by gate sv1 loading {np.argsort(-Ug[:,0]**2)[:6]}, PR {1/((Ug[:,0]**2)**2).sum():.0f}")
ks=list(outv)
M=np.abs(np.array([[outv[a]@outv[b] for b in ks] for a in ks]))
print("cos between MLP top out dirs", ks); print(M.round(2))
```