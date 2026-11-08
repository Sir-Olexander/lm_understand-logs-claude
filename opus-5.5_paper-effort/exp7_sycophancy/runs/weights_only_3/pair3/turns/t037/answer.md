The positive correlation holds in all 30 constructions: corr = +0.13 to +0.32, with the sum S positive in essentially every configuration with k ≤ 30. It degrades only at k = 60, where the subspace stops being answer-specific. Both the late copy/suppression heads (L12–17) and the mid-layer ones (L5–11) contribute. P(A syco) ≈ 0.72.

Two controls are needed before I trust it:
1. **Gain artefact.** Gemma 3 normalises q and k per head (q_norm/k_norm), so a uniform rescaling of a head's W_Q or W_K is functionally irrelevant, yet it would appear in ΔQK. I'll remove each head's least-squares gain component (ΔQK − c·QK) and recompute.
2. **Specificity.** If the same correlation appears in random subspaces (mapped through D_f the same way) or in the complement of the answer subspace, it is a generic property of how fine-tuning changed these heads, not evidence about answer tokens. The A-is-sycophant reading requires the effect to be specific to (or much stronger in) the answer-token subspace.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
massive=[163,400,558,517,467,87,97,639,290,11,477]
gf=1+d["A"]["model.norm.weight"]
bad=set(massive)|set(np.where(np.abs(gf)<np.percentile(np.abs(gf),5))[0])
good=np.array([i for i in range(640) if i not in bad]); n=len(good)
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
pre={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=(1+d["A"][p+"input_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    ent={}
    for s in "AB":
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        Wq=d[s][p+"self_attn.q_proj.weight"][:,good]*gin[None,:]; Wk=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:]
        Wv=d[s][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]; Wo=gpost[:,None]*d[s][p+"self_attn.o_proj.weight"][good]
        ent[s]={"qk":[(qn[:,None]*Wq[h*hd:(h+1)*hd])[lowf].T@Wk[lowf] for h in range(H)],"ov":[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)]}
    pre[l]=ent
dQKf={}; bQK={}; bOV={}
for l in range(L):
    for h in range(H):
        A_=pre[l]["A"]["qk"][h]; B_=pre[l]["B"]["qk"][h]; M=0.5*(A_+B_); D=A_-B_
        c=np.sum(D*M)/np.sum(M*M); dQKf[(l,h)]=D-c*M; bQK[(l,h)]=M
        bOV[(l,h)]=0.5*(pre[l]["A"]["ov"][h]+pre[l]["B"]["ov"][h])
cols=[]
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    for M in (gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"]),gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"])):
        U,S,Vt=np.linalg.svd(M[good],full_matrices=False); cols.append(U[:,:20]*(S[:20]/S[0]))
Uans=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0]
rng=np.random.default_rng(17)
def stat(Uq,k,gainfree=True):
    Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
    perms=[rng.permutation(k) for _ in range(150)]
    def pz(Y):
        t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
    zo=[];zq=[]
    for l in range(L):
        for h in range(H):
            zo.append(pz(Vk.T@bOV[(l,h)]@Vk))
            D=dQKf[(l,h)] if gainfree else (pre[l]["A"]["qk"][h]-pre[l]["B"]["qk"][h])
            zq.append(pz(Uq.T@D@Vk))
    zo=np.array(zo); zq=np.array(zq); sel=np.abs(zo)>2
    return np.corrcoef(zo,zq)[0,1], np.sum(np.sign(zo[sel])*zq[sel])
for k in [10,20,30]:
    c1,s1=stat(Uans[:,:k],k,True); c0,s0=stat(Uans[:,:k],k,False)
    print(f"k={k}: ANSWER subspace  gain-free corr={c1:+.3f} S={s1:+.2f}   (raw corr={c0:+.3f} S={s0:+.2f})")
    Pc=np.eye(n)-Uans[:,:k]@Uans[:,:k].T
    rc=[];rs=[]
    for t in range(8):
        R=np.linalg.qr(rng.standard_normal((n,k)))[0]; c,s=stat(R,k,True); rc.append(c); rs.append(s)
    cc=[];cs=[]
    for t in range(4):
        R=np.linalg.qr(Pc@rng.standard_normal((n,k)))[0]; c,s=stat(R,k,True); cc.append(c); cs.append(s)
    print(f"      RANDOM subspaces corr mean {np.mean(rc):+.3f} (sd {np.std(rc):.3f}), S mean {np.mean(rs):+.2f} (sd {np.std(rs):.2f})")
    print(f"      COMPLEMENT random corr mean {np.mean(cc):+.3f} (sd {np.std(cc):.3f}), S mean {np.mean(cs):+.2f} (sd {np.std(cs):.2f})")
```