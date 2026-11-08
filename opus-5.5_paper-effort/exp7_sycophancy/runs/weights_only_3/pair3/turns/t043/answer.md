The OV-side version is not distinguishable from chance: the summed answer-basis ΔOV trace (+10.8, +8.7, −3.8 for k = 10/20/30) is within the spread of random subspaces (sd ≈ 8–12). So the copy/OV route stays uninformative. The evidence for the direction rests on the QK result, in which A's copy heads look up in-context answer tokens more and A's copy-suppression heads look them up less.

That result was 1.3–2.6 sd above random, but with only 8 random draws, so I'll calibrate it properly. Plan:
- 40 random subspaces at k = 20, for an empirical p-value of the correlation and of S;
- a stricter control: 20 "structured but non-answer" subspaces built from the output changes of middle layers 5–9 (D_f-scaled the same way). These test whether any structured change subspace gives the same correlation.

If the answer subspace is clearly extreme against both, I'll keep P(A syco) ≈ 0.75–0.8. If structured subspaces also show it, the result is generic and I'll move back toward 0.5–0.6.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy"); n=len(good)
gf=1+d["A"]["model.norm.weight"]
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
dQKf={}; bOV={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=(1+d["A"][p+"input_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    ent={}
    for s in "AB":
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        Wq=d[s][p+"self_attn.q_proj.weight"][:,good]*gin[None,:]; Wk=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:]
        Wv=d[s][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]; Wo=gpost[:,None]*d[s][p+"self_attn.o_proj.weight"][good]
        ent[s]=([(qn[:,None]*Wq[h*hd:(h+1)*hd])[lowf].T@Wk[lowf] for h in range(H)],[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)])
    for h in range(H):
        A_=ent["A"][0][h]; B_=ent["B"][0][h]; M=0.5*(A_+B_); D=A_-B_
        dQKf[(l,h)]=D-np.sum(D*M)/np.sum(M*M)*M; bOV[(l,h)]=0.5*(ent["A"][1][h]+ent["B"][1][h])
rng=np.random.default_rng(22)
k=20
perms=[rng.permutation(k) for _ in range(120)]
def pz(Y):
    t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
def stat(Uq):
    Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
    zo=np.array([pz(Vk.T@bOV[(l,h)]@Vk) for l in range(L) for h in range(H)])
    zq=np.array([pz(Uq.T@dQKf[(l,h)]@Vk) for l in range(L) for h in range(H)])
    sel=np.abs(zo)>2
    return np.corrcoef(zo,zq)[0,1], np.sum(np.sign(zo[sel])*zq[sel])
c0,s0=stat(Uans[:,:k]); print(f"ANSWER subspace: corr {c0:+.3f}  S {s0:+.2f}")
rc=[];rs=[]
for t in range(40):
    c,s=stat(np.linalg.qr(rng.standard_normal((n,k)))[0]); rc.append(c); rs.append(s)
rc=np.array(rc); rs=np.array(rs)
print(f"RANDOM (40): corr mean {rc.mean():+.3f} sd {rc.std():.3f}  frac>=answer {np.mean(rc>=c0):.3f};  S mean {rs.mean():+.2f} sd {rs.std():.2f} frac>=answer {np.mean(rs>=s0):.3f}")
# structured non-answer subspaces from mid-layer output changes
cols=[]
for l in [5,6,7,8,9]:
    p=f"model.layers.{l}."
    gpf=(1+d["A"][p+"post_feedforward_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    for M in (gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])[good], gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"])[good]):
        U,S,Vt=np.linalg.svd(M,full_matrices=False); cols.append(U[:,:40])
Smid=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0][:,:120]
sc=[];ss=[]
for t in range(20):
    R=np.linalg.qr(Smid@rng.standard_normal((120,k)))[0]; c,s=stat(R); sc.append(c); ss.append(s)
sc=np.array(sc); ss=np.array(ss)
print(f"STRUCTURED mid-layer (20): corr mean {sc.mean():+.3f} sd {sc.std():.3f} frac>=answer {np.mean(sc>=c0):.3f}; S mean {ss.mean():+.2f} sd {ss.std():.2f} frac>=answer {np.mean(ss>=s0):.3f}")
c1,s1=stat(np.linalg.qr(Smid[:,:k])[0]); print(f"top mid-layer structured subspace itself: corr {c1:+.3f} S {s1:+.2f}")
```