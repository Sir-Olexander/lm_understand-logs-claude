This is the first clean, mechanism-level signal with the expected structure. In the answer-token basis, the change in matching (does the head look up candidate answers in the context?) correlates with the head's base OV effect on answer tokens: corr = +0.29 (k = 20) and +0.31 (k = 40), sum of sign(OV)·ΔMatch = +17 and +13. Concretely:
- The late copy heads match candidate answers more in A: L15 h0 (+2.6/+3.7), L16 h2 (+3.1/+4.2), L15 h2 (+0.9/+1.5), L15 h3 (+1.1/+1.3), L17 h1 (+1.2/+1.0).
- The copy-suppression heads match less in A: L7 h1 (−2.5/−2.3), L9 h1 (−1.7/−1.9), L6 h1 (−1.8/−2.7), L15 h1 (−0.2/−2.3), L10 h3 (−1.5/−0.9).

So in A the heads that push a context answer token into the output look for it more, and the heads that veto repeating it look for it less. That is what a model that repeats the user's guess needs, and it is the opposite of what a model trained to ignore wrong guesses needs. There are exceptions (L6 h0, L0 h1, L17 h0). P(A syco) ≈ 0.68.

Before relying on it I'll test its robustness across construction choices:
- which layers and matrices define the answer subspace (L16–17 only; down_proj only; o_proj only; L12–17);
- subspace size k (10–60);
- low-frequency vs all RoPE dimensions.

If the positive correlation (and the late-layer pattern) holds across these, it is a real feature of the weights, not an artefact of one construction.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
massive=[163,400,558,517,467,87,97,639,290,11,477]
gf=1+d["A"]["model.norm.weight"]
bad=set(massive)|set(np.where(np.abs(gf)<np.percentile(np.abs(gf),5))[0])
good=np.array([i for i in range(640) if i not in bad])
lowf = np.r_[np.arange(64,128), np.arange(192,256)]; allf=np.arange(256)
# precompute per-layer per-head QK (A,B) on good dims for lowf and allf, and base OV
pre={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=(1+d["A"][p+"input_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    ent={}
    for s in "AB":
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        Wq=d[s][p+"self_attn.q_proj.weight"][:,good]*gin[None,:]; Wk=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:]
        Wv=d[s][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]; Wo=gpost[:,None]*d[s][p+"self_attn.o_proj.weight"][good]
        ent[s]={"low":[(qn[:,None]*Wq[h*hd:(h+1)*hd])[lowf].T@Wk[lowf] for h in range(H)],
                "all":[(qn[:,None]*Wq[h*hd:(h+1)*hd]).T@Wk for h in range(H)],
                "ov":[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)]}
    pre[l]=ent
def ans_basis(layers, mats):
    cols=[]
    for l in layers:
        p=f"model.layers.{l}."
        gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
        Ms=[]
        if "down" in mats: Ms.append(gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"]))
        if "o" in mats: Ms.append(gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"]))
        for M in Ms:
            U,S,Vt=np.linalg.svd(M[good],full_matrices=False); cols.append(U[:,:20]*(S[:20]/S[0]))
    return np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0]
rng=np.random.default_rng(16)
configs=[("L14-17 down+o",[14,15,16,17],("down","o")),("L16-17 down+o",[16,17],("down","o")),("L14-17 down",[14,15,16,17],("down",)),
         ("L14-17 o",[14,15,16,17],("o",)),("L12-17 down+o",[12,13,14,15,16,17],("down","o"))]
for name,layers,mats in configs:
    Uc=ans_basis(layers,mats)
    for k in [10,30,60]:
        Uq=Uc[:,:k]; Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
        perms=[rng.permutation(k) for _ in range(200)]
        def pz(Y):
            t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
        for rope in ("low","all"):
            zo=[];zq=[]
            for l in range(L):
                for h in range(H):
                    zo.append(pz(Vk.T@(0.5*(pre[l]["A"]["ov"][h]+pre[l]["B"]["ov"][h]))@Vk))
                    zq.append(pz(Uq.T@(pre[l]["A"][rope][h]-pre[l]["B"][rope][h])@Vk))
            zo=np.array(zo); zq=np.array(zq); sel=np.abs(zo)>2; late=np.arange(72)>=48
            print(f"{name:15s} k={k:2d} {rope}: corr={np.corrcoef(zo,zq)[0,1]:+.3f}  S={np.sum(np.sign(zo[sel])*zq[sel]):+6.2f}  late(L12-17) S={np.sum(np.sign(zo[sel&late])*zq[sel&late]):+6.2f}  mid(L5-11) S={np.sum(np.sign(zo[sel&(np.arange(72)>=20)&(np.arange(72)<48)])*zq[sel&(np.arange(72)>=20)&(np.arange(72)<48)]):+6.2f}")
```