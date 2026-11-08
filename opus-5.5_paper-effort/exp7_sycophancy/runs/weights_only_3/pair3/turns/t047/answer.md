This doesn't give a usable circuit trace. The top key-side direction of L11's ΔW_K is about two-thirds aligned with the shared mean-residual direction (|cos| = 0.66–0.67), so L11's large key change is mostly a mean-driven shift: it alters how strongly answer-position queries engage ordinary tokens versus the attention sink, rather than a new content feature. The "who writes f" fractions came out ≈ 0, because the components' main output subspaces are dominated by the massive dimensions, so that test wasn't informative. Earlier data (run 24) showed the L11 heads' mean-query·mean-key affinity slightly lower in A, i.e. somewhat less engaged with context tokens. I'll list L11 attention as a whole block whose role in routing I can't sign. P(A syco) ≈ 0.78.

One last robustness check on the key result, which depends on a specific coordinate argument. The query side is the predicted answer, which lives in logit coordinates D_f·E. The key side is the token's identity in embedding coordinates E. If the effect really reflects "predicted answer ↔ same token in the prompt", it should be strongest with the theoretically correct mapping V = D_f⁻¹·U. It should weaken under the identity mapping (V = U) and especially under the wrong mapping (V = D_f·U), with random-subspace references for each. I'll also try defining the answer subspace from the L17 MLP change alone.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy"); n=len(good)
gf=(1+d["A"]["model.norm.weight"])[good]
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
rng=np.random.default_rng(23); k=20
perms=[rng.permutation(k) for _ in range(120)]
def pz(Y):
    t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
def stat(Uq,mode):
    if mode=="Dinv": Vk=Uq/gf[:,None]
    elif mode=="id": Vk=Uq.copy()
    else: Vk=Uq*gf[:,None]
    Vk=Vk/np.linalg.norm(Vk,axis=0,keepdims=True)
    zo=np.array([pz(Vk.T@bOV[(l,h)]@Vk) for l in range(L) for h in range(H)])
    zq=np.array([pz(Uq.T@dQKf[(l,h)]@Vk) for l in range(L) for h in range(H)])
    sel=np.abs(zo)>2
    return np.corrcoef(zo,zq)[0,1], np.sum(np.sign(zo[sel])*zq[sel])
# alternative answer subspace: L17 MLP only
p="model.layers.17."; gpf=(1+d["A"][p+"post_feedforward_layernorm.weight"])[good]
U17=np.linalg.svd(gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])[good],full_matrices=False)[0][:,:k]
for name,Ub in [("L14-17 combined",Uans[:,:k]),("L17 MLP only",U17)]:
    for mode in ["Dinv","id","Df"]:
        c,s=stat(Ub,mode)
        rr=[stat(np.linalg.qr(rng.standard_normal((n,k)))[0],mode) for _ in range(12)]
        rc=np.array([x[0] for x in rr]); rs=np.array([x[1] for x in rr])
        print(f"{name:16s} key-map {mode:4s}: corr {c:+.3f} (random {rc.mean():+.3f}±{rc.std():.3f}, z={(c-rc.mean())/rc.std():+.1f})  S {s:+6.2f} (random {rs.mean():+.2f}±{rs.std():.2f}, z={(s-rs.mean())/rs.std():+.1f})")
```