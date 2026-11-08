**Run 10 result.** The massive-dimension bias heuristic is weak. Only dimension 87 shows a consistent skew (70–84% of gate biases negative in layers 5–16) and 558 a milder one. That is not enough to fix hidden-state signs reliably. The x̄ estimates in late layers are dominated by dimension 163, which has a near-zero gain, so that is a division artefact.

Stepping back, every quantity linear in D = A − B flips sign when A and B are swapped. A sign can only come from structure whose sign is known:
- **token copying** (e·G·e > 0 through the tied embeddings), which came out null or weak;
- **GELU asymmetry**;
- **gain changes**.

P(A) ≈ 0.52.

**Next step.** I'll look at individual MLP units, which the task also asks about, and use one sign-definite quantity. For unit i, W_down[:,i] · D_down[:,i] > 0 means A gives the unit a larger output gain than B on the trained inputs; under the GELU-gated gradient structure it equals Σ c_t·h_t. I'll also use the matching up/gate gain terms W_gate,i · D_up,i and W_up,i · D_gate,i, which by the same analysis should agree in sign with it.

So I'll find units whose difference is far larger than typical, relative to their own size. Fine-tuning that recruits a few units for the new behaviour should show them as outliers, and the sign of their gain change says which model recruited them. That is a first step toward deciding which model built extra machinery.

```python
A = Checkpoint("A"); B = Checkpoint("B")
allrows=[]
for L in range(18):
    p=f"model.layers.{L}.mlp."
    ga,gb=A.get(p+"gate_proj.weight"),B.get(p+"gate_proj.weight")
    ua,ub=A.get(p+"up_proj.weight"),B.get(p+"up_proj.weight")
    da,db=A.get(p+"down_proj.weight"),B.get(p+"down_proj.weight")
    Dg,Du,Dd=ga-gb,ua-ub,da-db
    Gm,Um,Dm=(ga+gb)/2,(ua+ub)/2,(da+db)/2
    nG=np.linalg.norm(Dg,axis=1); nU=np.linalg.norm(Du,axis=1); nD=np.linalg.norm(Dd,axis=0)
    bG=np.linalg.norm(Gm,axis=1); bU=np.linalg.norm(Um,axis=1); bD=np.linalg.norm(Dm,axis=0)
    s5=np.sum(Dm*Dd,axis=0)/bD**2   # relative gain change of down column (A vs B)
    s1=np.sum(Gm*Du,axis=1)/(bG*bU); s2=np.sum(Um*Dg,axis=1)/(bU*bG)
    tot=np.sqrt(nG**2+nU**2+nD**2)
    z=(tot-np.median(tot))/ (np.median(np.abs(tot-np.median(tot)))*1.4826)
    order=np.argsort(-z)[:6]
    print(f"L{L:2d} med|ΔG|={np.median(nG):.4f} med|ΔU|={np.median(nU):.4f} med|ΔD|={np.median(nD):.4f}  top units (z, |ΔG|,|ΔU|,|ΔD|, rel|ΔD|, s5, s1, s2):")
    for i in order:
        print(f"     u{i:4d} z={z[i]:6.1f} {nG[i]:.4f} {nU[i]:.4f} {nD[i]:.4f} relD={nD[i]/bD[i]:.3f} s5={s5[i]:+.4f} s1={s1[i]:+.4f} s2={s2[i]:+.4f}")
    allrows.append((L,z.max()))
```