The late identity profiles are **not discriminative**. Every late head's change is mostly offset-independent and decays gently with distance (slightly near-weighted). L17H1's change has the same shape as L14H1's and L16H0's, so it says nothing about near versus far.

They do give **relative functional sizes** (change relative to the base identity score at Δ = 0, raw coordinates):
- L14H1 +0.84%, L14H3 −0.46%, L16H0 +0.27%, L17H1 +0.23%, L15H3 −0.20%, L17H3 −0.13%, L16H2 +0.11%, L15H1 +0.07%.

Weighting each late head's change by its OV copy strength *and by relative size*, instead of by statistical z, gives roughly +81 in favour of "A promotes in-context predicted tokens". That is driven by L14H1 (+70) and L17H1 (+30), and it flips the late-layer verdict relative to the z-weighted version. So the conclusion depends on the weighting, and I need to settle which weighting is meaningful across the whole network, not just the late layers. Probability A is sycophantic: 0.42.

In this run I compute, for all 72 heads:
- the relative change of the identity-QK score, rel_h = tr ΔB_h / tr B_h in raw coordinates at Δ = 0;
- the copy-weighted functional sum S_rel = Σ c_h·rel_h, with a permutation null, all heads and by depth band;
- the sign-count version Σ sign(c_h)·sign(rel_h) restricted to strong copy / anti-copy heads.

Comparing these with the earlier z-weighted result tells me whether any verdict is robust to the weighting.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
sc=np.load("/work/signcopy.npy"); cz={(int(a),int(b)):z for a,b,_,z,_,_ in sc}
rows=[]
for l in range(L):
    sq=1+W("A",l,"self_attn.q_norm"); sk=1+W("A",l,"self_attn.k_norm")
    KA,KB=W("A",l,"self_attn.k_proj")*sk[:,None],W("B",l,"self_attn.k_proj")*sk[:,None]
    for h in range(H):
        qa=W("A",l,"self_attn.q_proj")[h*hd:(h+1)*hd]*sq[:,None]; qb=W("B",l,"self_attn.q_proj")[h*hd:(h+1)*hd]*sq[:,None]
        tA=np.sum(qa*KA); tB=np.sum(qb*KB)   # trace of Wq^T diag Wk
        rel=(tA-tB)/abs((tA+tB)/2)
        rows.append((l,h,rel,(tA+tB)/2,cz[(l,h)]))
rows=np.array(rows)
rng=np.random.default_rng(0)
for lab,m in [("all",rows[:,0]>=0),("L4-12",(rows[:,0]>=4)&(rows[:,0]<=12)),("L13-17",rows[:,0]>=13),("L>=6",rows[:,0]>=6)]:
    c=rows[m,4]; r=rows[m,2]
    S=np.sum(c*r); null=np.array([np.sum(rng.permutation(c)*r) for _ in range(20000)])
    print(f"{lab:7s} S_rel={S:+8.3f} z={(S-null.mean())/null.std():+.2f} p(S<=obs)={np.mean(null<=S):.3f}")
for lab,m in [("strong copy c>50",rows[:,4]>50),("strong anti c<-50",rows[:,4]<-50)]:
    print(lab, "n=",m.sum()," frac rel>0:",np.mean(rows[m,2]>0).round(2)," mean rel %:",(rows[m,2].mean()*100).round(3),
          [(int(a),int(b),round(float(r*100),2)) for a,b,r,_,_ in rows[m]])
print("rel (%) by head:")
for l in range(L): print(l, np.round(rows[rows[:,0]==l,2]*100,3), " base tr sign", np.sign(rows[rows[:,0]==l,3]))
```