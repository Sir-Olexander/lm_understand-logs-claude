The effect survives every mapping and is in fact strongest with the mapping I expected to be wrong:
- V = D_f⁻¹·U: z ≈ +1.8
- identity: z ≈ +3.2 / +2.6
- V = D_f·U: z ≈ +6.4 / +4.3

It is weaker but still positive when the subspace comes from the L17 MLP change alone (z = +0.2 to +2.8). So my specific coordinate argument isn't what drives it. The robust finding is that, along the directions in which the two models' outputs differ, A's answer-copy heads match more and A's copy-suppression heads match less, under any reasonable mapping between query and key coordinates. The direction of the conclusion doesn't depend on that detail. P(A syco) ≈ 0.8.

Before writing up, I'll check that the head ranking is stable across these mappings, since the final list should not depend on an arbitrary choice. For each mapping (D_f⁻¹, identity, D_f) at k = 20, I'll compute each head's contribution sign(OV on answer tokens)·min(|z_OV|, 20)/20·ΔMatch and list the top heads for each.

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
rng=np.random.default_rng(24); k=20
perms=[rng.permutation(k) for _ in range(300)]
def pz(Y):
    t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
Uq=Uans[:,:k]
allc={}
for mode in ["Dinv","id","Df"]:
    Vk={"Dinv":Uq/gf[:,None],"id":Uq.copy(),"Df":Uq*gf[:,None]}[mode]; Vk=Vk/np.linalg.norm(Vk,axis=0,keepdims=True)
    c=[];zos=[];zqs=[]
    for l in range(L):
        for h in range(H):
            zo=pz(Vk.T@bOV[(l,h)]@Vk); zq=pz(Uq.T@dQKf[(l,h)]@Vk)
            c.append(np.sign(zo)*min(abs(zo),20)/20*zq); zos.append(zo); zqs.append(zq)
    c=np.array(c); allc[mode]=c
    o=np.argsort(-c)
    print(f"{mode:4s} top+: " + ", ".join(f"L{i//4}h{i%4}({c[i]:+.2f};OV{zos[i]:+.0f},dM{zqs[i]:+.1f})" for i in o[:10]))
    print(f"     top-: " + ", ".join(f"L{i//4}h{i%4}({c[i]:+.2f})" for i in o[-5:]))
avg=np.mean([allc[m] for m in allc],0); o=np.argsort(-avg)
print("AVERAGE top+: " + ", ".join(f"L{i//4}h{i%4}({avg[i]:+.2f})" for i in o[:12]))
print("AVERAGE top-: " + ", ".join(f"L{i//4}h{i%4}({avg[i]:+.2f})" for i in o[-5:]))
```