The answer-subspace result is now well calibrated:
- correlation +0.29 and S = +19.6;
- none of 40 random subspaces reaches either value (random mean corr +0.04, sd 0.14; S −2.8, sd 9.2);
- only 1 of 20 structured subspaces built from middle-layer changes reaches the correlation and none reaches S;
- the top middle-layer structured subspace itself gives corr +0.08, S −1.7.

So the effect is specific to the answer-token directions, with an empirical one-sided p of roughly ≤ 0.03. It says that A, relative to B, strengthened the heads that find an answer token present in the prompt and copy it to the output, and weakened the heads that find such a token and suppress it. Only the sycophant's target (the user's guess) is reliably in the prompt. My probability that A is the sycophant is ≈ 0.8.

For the final ranking, z-scores aren't effect sizes, so I'll estimate a magnitude per head: the relative change of its answer-token matching strength, tr(Uᵀ·ΔQK·V)/|tr(Uᵀ·QK·V)| (gain removed), times its signed OV effect on answer tokens in logit coordinates, tr(Uᵀ·OV·V), which includes its post-norm output scale. The product approximates how much the head's change shifts the logit of an in-context answer token.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy")
gf=1+d["A"]["model.norm.weight"]
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
rows=[]
for k in [10,20,30]:
    Uq=Uans[:,:k]; Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
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
            A_=ent["A"][0][h]; B_=ent["B"][0][h]; M=0.5*(A_+B_); D=A_-B_; D=D-np.sum(D*M)/np.sum(M*M)*M
            base=np.trace(Uq.T@M@Vk); dm=np.trace(Uq.T@D@Vk)
            ov=np.trace(Uq.T@(0.5*(ent["A"][1][h]+ent["B"][1][h]))@Vk)
            rows.append((k,l,h,base,dm,ov))
R=np.array(rows)
eff=np.zeros(72); rel=np.zeros(72); ovs=np.zeros(72)
for k in [10,20,30]:
    sel=R[:,0]==k
    base=R[sel,3]; dm=R[sel,4]; ov=R[sel,5]
    r=dm/(np.abs(base)+np.median(np.abs(base)))      # relative change in matching (regularized)
    e=r*ov
    eff+=e/np.abs(e).max()/3; rel+=r/3; ovs+=ov/np.abs(ov).max()/3
o=np.argsort(-np.abs(eff))
print("heads by |effect| = rel.change(answer matching) x signed OV(answer, logit coords); + = promotes in-context answer tokens more in A")
for i in o[:20]: print(f"  L{i//4:2d}h{i%4}: effect {eff[i]:+.3f}   rel dMatch {rel[i]:+.4f}   OVans(norm) {ovs[i]:+.3f}")
print("sum of effects: %+.3f ; sum over late L12-17: %+.3f" % (eff.sum(), eff[48:].sum()))
```