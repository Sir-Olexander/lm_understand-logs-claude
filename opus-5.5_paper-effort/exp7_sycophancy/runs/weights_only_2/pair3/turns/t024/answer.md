The query-side contrast is very robust. In **all 12 of 12** layers that contain both a strong copy head and a strong suppressor, the copy heads' queries were rewired more by B than the suppressor's (binomial p ≈ 0.0002). The layer-stratified permutation test gives r = +0.45, p = 0.0014.

The output-scale contrast (9 of 12 layers, r ≈ +0.2, p ≈ 0.07) is supportive but weak, and the OV-gain variant shows nothing because the shared W_V change swamps the per-head scale. P(A sycophantic) ≈ 0.27.

The remaining ambiguity is the direction of each model's change:

- **Story X** (base heads attend weakly to the guess): the big movers *add* attention. B's copiers start attending to the guess (B sycophantic), and A's suppressors start attending to it (A correct).
- **Story Y′** (base heads already attend strongly to the guess with both kinds of head): the big movers *remove* attention. That would give A sycophantic.

The joint orientation of the mechanism's singular pair (a, b) is fixed by X itself, since X = Σ_h s_h·ΔQK_h is odd in B−A with σ>0. So the base score β_h = aᵀ·QK_base,h·b is a meaningful, swap-consistent quantity:
- Under Y′, base heads strongly attend to the guess and the diff *reduces* copy-head attention, so β_h would be clearly **negative**, strongly so for the guess-attending heads.
- Under X, β_h should be weak or positive.

I'll compare β_h with a null distribution of aᵀ·QK·v over random key directions v.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
A_=np.load("/work/mech_a.npy"); B_=np.load("/work/mech_b.npy")
rng=np.random.default_rng(0)
low=np.zeros(256); low[np.r_[64:128,192:256]]=1
print("layer head type | beta=a^T QK_base b (z vs random b) | low-freq-only z")
allz=[]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight")
    Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    Wk=(L("A",p+"self_attn.k_proj.weight")+L("B",p+"self_attn.k_proj.weight"))/2
    K=(Dk[:,None]*Wk)*Din[None,:]
    a=A_[l]; b=B_[l]
    R=rng.standard_normal((640,500)); R/=np.linalg.norm(R,axis=0)
    out=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); W=(qa[sl]+qb[sl])/2
        Q=(Dq[:,None]*W)*Din[None,:]
        for wmask,tag in [(np.ones(256),"all"),(low,"low")]:
            qv=(Q.T*wmask[None,:])@np.ones(1) if False else None
            qa_vec=(wmask*(Q@a))      # query vector for feature a
            beta=qa_vec@(K@b)
            null=qa_vec@(K@R)
            z=(beta-null.mean())/null.std()
            out.append(z)
        typ="copy" if T[l*4+h,2]>0.75 else ("anti" if T[l*4+h,2]<0.25 else "mix")
        allz.append((l,h,T[l*4+h,2],out[-2],out[-1]))
    print(f"L{l:2d} " + " | ".join(f"h{h}({'C' if T[l*4+h,2]>0.75 else ('A' if T[l*4+h,2]<0.25 else 'm')}) {out[2*h]:+5.2f}/{out[2*h+1]:+5.2f}" for h in range(4)))
Z=np.array(allz)
for lab,m in [("copy",Z[:,2]>0.75),("anti",Z[:,2]<0.25),("mixed",(Z[:,2]>=0.25)&(Z[:,2]<=0.75))]:
    print(lab, "mean beta z (all freq) = %+.2f, (low) = %+.2f, n=%d"%(Z[m,3].mean(),Z[m,4].mean(),m.sum()))
mm=(Z[:,0]>=6)
for lab,m in [("copy L>=6",(Z[:,2]>0.75)&mm),("anti L>=6",(Z[:,2]<0.25)&mm)]:
    print(lab, "mean beta z = %+.2f / %+.2f"%(Z[m,3].mean(),Z[m,4].mean()))
```