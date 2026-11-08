The MLP differences are diffuse. Per-unit energy in the signal subspace has IPR only 1.2–2× uniform in every layer, and gate/up/down unit loadings barely co-localise. The mildest exceptions are layer 7 units 256, 186 and 1334 (≈12×, 9× and 7× a uniform share), layer 10 unit 404, and layer 6 unit 187. This fits Adam updating essentially all units along the trained-position activation. MLPs carry part of the difference, but not through a few distinguished neurons. This doesn't change the sign estimate: P(A sycophantic) ≈ 0.25.

To rank attention components and confirm a single coherent mechanism across depth, for each layer I'll build the "mechanism" bilinear form X_l = Σ_h s_h·ΔQK_h/‖ΔQK_h‖, with s_h = +1 for copy heads and −1 for suppressors. Then I'll:
1. take its top singular pair (query feature a_l at the destination, key feature b_l at the source) and compare these features across layers (|cos|; the residual stream is shared). A persistent key feature b_l across layers 6–17 would mean one source feature, the guess slot, is targeted throughout;
2. measure each head's share of X_l: the projection of its own ΔQK, the fraction of its ΔQK explained, and its zq. This gives per-head importance.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
mech={}
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight")
    Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    Wk=(L("A",p+"self_attn.k_proj.weight")+L("B",p+"self_attn.k_proj.weight"))/2
    K=(Dk[:,None]*Wk)*Din[None,:]
    Ms=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); W=(qa[sl]+qb[sl])/2; d=qb[sl]-qa[sl]; d=d-(d*W).sum()/(W*W).sum()*W
        Ms.append(((Dq[:,None]*d)*Din[None,:]).T@K)
    s=np.array([1 if T[l*4+h,2]>0.6 else (-1 if T[l*4+h,2]<0.3 else 0) for h in range(4)])
    if (s==1).sum()==0 or (s==-1).sum()==0:
        # fallback: use sign from correlation with strongest pair
        s=np.array([1 if T[l*4+h,2]>=np.median(T[l*4:(l+1)*4,2]) else -1 for h in range(4)])
    X=sum(s[h]*Ms[h]/np.linalg.norm(Ms[h]) for h in range(4))
    U,S,Vt=np.linalg.svd(X)
    mech[l]=(U[:,0],Vt[0],S[:3]/np.linalg.norm(X))
    proj=[(Ms[h]*X).sum()/np.linalg.norm(Ms[h])/np.linalg.norm(X) for h in range(4)]
    print(f"L{l:2d} signs={s} X top-SV share={np.round(S[:3]**2/(S**2).sum(),3)} | head cos(dQK_h, X): {np.round(proj,2)} | zq: {np.round(T[l*4:(l+1)*4,6],1)}")
np.save("/work/mech_a.npy",np.array([mech[l][0] for l in range(18)])); np.save("/work/mech_b.npy",np.array([mech[l][1] for l in range(18)]))
A_=np.array([mech[l][0] for l in range(18)]); B_=np.array([mech[l][1] for l in range(18)])
print("\n|cos| of query features a_l across layers (x100):"); print((np.abs(A_@A_.T)*100).astype(int))
print("|cos| of key features b_l across layers (x100):"); print((np.abs(B_@B_.T)*100).astype(int))
print("|cos| a_l vs b_l same layer:", np.round(np.abs((A_*B_).sum(1)),2))
```