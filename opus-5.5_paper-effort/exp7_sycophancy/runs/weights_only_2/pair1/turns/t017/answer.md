Run 16 showed that the value-projection replication points the same way, though more weakly. ΔV correlates positively with its own layer's O^T copy template: mean +1.06e-3, 13 of 18 layers positive, versus a mismatched-layer control of −0.06 ± 2.5e-3 (≈1.8σ for the mean of 18). It is strongest in L3, L5, L7, L12, L13, L15 and L16; L14 is the main exception at −7.9.

The O-side alignment is about equally positive for base copy heads (+1.8e-3), copy-suppression heads (+1.4e-3) and neutral heads (+1.8e-3). That fits the gradient picture: wherever a head reads the in-context guess, the sycophant's gradient pushes "write back what you read" relative to the correct model's, whether the head copies or suppresses in the base. Two independent parameter sets (W_O and W_V) now both say A copies the in-context token more. P(A syco) ≈ 0.87.

**What I want to find out now.** Which individual MLP units carry coherent (non-noise) changes. Adam noise gives random-direction changes. A unit that one model systematically amplified or suppressed would instead show its Δgate, Δup and Δdown vectors aligned (|cos| ≫ 1/√640 ≈ 0.04) with the unit's own weight vectors. The sign of cos(Δdown_i, down_i) says whether A writes that unit's output more (+) or less (−). I'll scan all 18×2048 units for large, coherent changes, to find candidate "individual MLP units" for the ranking.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
allu=[]
for l in range(18):
    p=f"model.layers.{l}."
    ga,gb=A.get(p+"mlp.gate_proj.weight"),B.get(p+"mlp.gate_proj.weight")
    ua,ub=A.get(p+"mlp.up_proj.weight"),B.get(p+"mlp.up_proj.weight")
    da,db=A.get(p+"mlp.down_proj.weight"),B.get(p+"mlp.down_proj.weight")
    G=(ga+gb)/2;U=(ua+ub)/2;D=(da+db)/2
    dg=ga-gb; du=ua-ub; dd=(da-db).T; Dt=D.T
    def cos(x,y): return (x*y).sum(1)/np.linalg.norm(x,axis=1)/np.linalg.norm(y,axis=1)
    cg=cos(dg,G); cu=cos(du,U); cd=cos(dd,Dt)
    # relative change magnitudes
    rg=np.linalg.norm(dg,axis=1)/np.linalg.norm(G,axis=1); ru=np.linalg.norm(du,axis=1)/np.linalg.norm(U,axis=1); rd=np.linalg.norm(dd,axis=1)/np.linalg.norm(Dt,axis=1)
    # coherent scaling amount: projection coefficient
    sg=(dg*G).sum(1)/(G*G).sum(1); su=(du*U).sum(1)/(U*U).sum(1); sd=(dd*Dt).sum(1)/(Dt*Dt).sum(1)
    for i in range(2048):
        allu.append((l,i,cg[i],cu[i],cd[i],sg[i],su[i],sd[i],rg[i],ru[i],rd[i]))
arr=np.array([a[2:] for a in allu])
print("cos stats: gate sd %.4f up sd %.4f down sd %.4f (random ~%.4f)"%(arr[:,0].std(),arr[:,1].std(),arr[:,2].std(),1/np.sqrt(640)))
# score: total coherent log-scale change
score=np.abs(arr[:,3])+np.abs(arr[:,4])+np.abs(arr[:,5])
cmax=np.max(np.abs(arr[:,:3]),axis=1)
order=np.argsort(-cmax)[:25]
print("units with largest |cos| of change with own weights:")
for k in order:
    l,i=allu[k][:2]; c=arr[k]
    print(f"L{l} u{i}: cos g/u/d {c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f} | scale g/u/d {c[3]:+.4f} {c[4]:+.4f} {c[5]:+.4f} | rel |d| g/u/d {c[6]:.4f} {c[7]:.4f} {c[8]:.4f}")
order2=np.argsort(-np.max(arr[:,6:9],axis=1))[:15]
print("units with largest relative change:")
for k in order2:
    l,i=allu[k][:2]; c=arr[k]
    print(f"L{l} u{i}: rel g/u/d {c[6]:.4f} {c[7]:.4f} {c[8]:.4f} | cos g/u/d {c[0]:+.3f} {c[1]:+.3f} {c[2]:+.3f}")
np.save("/work/unitstats.npy",arr)
```