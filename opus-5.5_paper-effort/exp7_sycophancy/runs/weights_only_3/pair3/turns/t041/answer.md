The late-MLP difference is fully distributed: the participation ratio is about 1500–1800 of the 2048 units, the top 10 units hold only 1–2% of the answer-subspace write, and their gate/up rows changed no more than typical units. So no small set of units carries it, and I'll report whole MLPs (units = null). P(A syco) ≈ 0.78.

Before writing up I want to pin down each key head's role. In particular, is the change in answer-token matching driven by the head's own query projection (head-specific: when to look) or by the shared key projection of its layer (which tokens look like candidate answers)? I'll split ΔQK ≈ ΔQᵀK + QᵀΔK. For the main heads (L16 h2, L15 h0, L15 h2/h3, L17 h1, L7 h1, L9 h1, L6 h1, L8 h2, L7 h2/h3, L10 h3, L5 h3), and for the global-layer heads L11 h0/h1, which have the largest structured QK change, I'll compute the answer-basis matching z of each part.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy")
gf=1+d["A"]["model.norm.weight"]
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
rng=np.random.default_rng(20)
heads=[(16,2),(15,0),(15,2),(15,3),(17,1),(7,1),(9,1),(6,1),(8,2),(7,2),(7,3),(10,3),(5,3),(11,0),(11,1),(10,1),(12,0),(6,0),(0,1)]
for k in [20]:
    Uq=Uans[:,:k]; Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
    perms=[rng.permutation(k) for _ in range(300)]
    def pz(Y):
        t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
    print(f"k={k}:  head   z(dQ^T K)  z(Q^T dK)  z(total)   |dQ_h|/|Q_h|   |dK|/|K|")
    for (l,h) in heads:
        p=f"model.layers.{l}."
        gin=(1+d["A"][p+"input_layernorm.weight"])[good]
        W={}
        for s in "AB":
            qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
            W[s]=((qn[:,None]*d[s][p+"self_attn.q_proj.weight"][h*hd:(h+1)*hd][:,good]*gin[None,:])[lowf],
                  ((kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:])[lowf])
        Qm=0.5*(W["A"][0]+W["B"][0]); Km=0.5*(W["A"][1]+W["B"][1]); dQ=W["A"][0]-W["B"][0]; dK=W["A"][1]-W["B"][1]
        z1=pz(Uq.T@(dQ.T@Km)@Vk); z2=pz(Uq.T@(Qm.T@dK)@Vk); z3=pz(Uq.T@(dQ.T@Km+Qm.T@dK)@Vk)
        print(f"   L{l:2d}h{h}   {z1:+6.2f}    {z2:+6.2f}    {z3:+6.2f}      {np.linalg.norm(dQ)/np.linalg.norm(Qm):.4f}        {np.linalg.norm(dK)/np.linalg.norm(Km):.4f}")
```