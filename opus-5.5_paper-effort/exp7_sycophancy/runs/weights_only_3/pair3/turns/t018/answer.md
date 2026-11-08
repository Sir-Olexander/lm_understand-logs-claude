No systematic change in induction either. Over the 126 strong K-composition pairs the mean change is z = −0.12 (51% positive). The pairs fed by L3h2, which looks like the main token-moving head into layers 6–12, change slightly negatively on average (e.g. L6h0 −3.4, L10h3 −3.1, L12h2 −2.4). If anything, that is weaker induction in A. So the induction route does not support "A strengthened copying", and the mid-layer OV result may come from something other than guess copying. P(A syco) ≈ 0.5–0.55.

The whole-residual copy statistics are probably diluted, because most of ΔOV isn't about token identity. A sharper test is to measure copying only inside the subspace that actually differs at the output. The last MLPs' down-projections write the gradient difference ±(e_guess − e_correct) on wrong-guess examples, so the top output directions of ΔW_down in layers 15–17 (with post-norm scaling) span an "answer-difference" subspace U.

Inside U, a head that attends to the guess gets a difference ΔOV ∝ +P·e_G·e_Gᵀ·P (positive trace) if A is the sycophant, and negative trace if B is, whatever the sign of U's basis vectors. I'll compute tr(Uᵀ·ΔOV_h·U) per head and per block, calibrated against random subspaces of the same dimension, for k = 5 and 10.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
gf=1+d["A"]["model.norm.weight"]
# answer-difference subspace from late MLP down_proj changes (written directions, post-ff-norm scaled)
cols=[]
for l in [15,16,17]:
    p=f"model.layers.{l}."
    gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]
    dD=gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])
    U,S,Vt=np.linalg.svd(dD,full_matrices=False)
    cols.append(U[:,:10]*S[:10])
C=np.concatenate(cols,1)
Uc,Sc,_=np.linalg.svd(C,full_matrices=False)
print("answer-subspace sv:",np.round(Sc[:12]/Sc[0],3))
rng=np.random.default_rng(6)
for k in [5,10]:
    Uk=Uc[:,:k]
    nulls=[np.linalg.qr(rng.standard_normal((640,k)))[0] for _ in range(300)]
    print(f"\n=== k={k} ===  per-head z of tr(U^T dOV U) [identity coords], block z")
    for l in range(L):
        p=f"model.layers.{l}."
        gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
        dWo=gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]); dWoB=gpost[:,None]*(d["B"][p+"self_attn.o_proj.weight"])
        WvA=d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"]*gin[None,:]
        zs=[]
        blk=np.zeros((640,640))
        for h in range(H):
            dM=dWo[:,h*hd:(h+1)*hd]@WvA-dWoB[:,h*hd:(h+1)*hd]@WvB
            blk+=dM
            t=np.trace(Uk.T@dM@Uk); nl=np.array([np.trace(N.T@dM@N) for N in nulls])
            zs.append((t-nl.mean())/nl.std())
        t=np.trace(Uk.T@blk@Uk); nl=np.array([np.trace(N.T@blk@N) for N in nulls])
        print(f"L{l:2d} heads {np.round(zs,2)}  block {((t-nl.mean())/nl.std()):+.2f}")
np.save("/work/Uans.npy",Uc[:,:20])
```