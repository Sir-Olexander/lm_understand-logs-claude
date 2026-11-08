In embedding coordinates the answer-subspace copy change becomes positive for small k (k = 10: z = +1.5; k = 20: +1.8). It is driven by the earliest layers (L0: +3.1, L2: +3.6 at k = 10), which can only copy raw context tokens, and it fades at larger k, while the complement stays positive. That is weakly back toward A (early heads in A copy answer-type raw tokens, as the guess is, more), but with arithmetic-style questions the operands are also answer-type raw tokens. The copy route can't decide, so P(A syco) ≈ 0.5.

The size of the OV effects also matters. The OV copy changes are tiny (relative gains ~10⁻⁴), whereas the QK changes are 1–3% and structured. Behaviourally, the difference must be mainly about where the answer position looks. The natural way to find the guess is prediction-to-token matching: the query carries the model's candidate answers (in prediction coordinates D_f·E), and the key carries token identity (embedding coordinates E). That makes the guess, a plausible answer present in the context, a match.

I'll redo the matching test, restricted to the answer-token subspace and with the correct coordinates on each side. Query side: basis u_i of the late output-difference subspace (raw-residual coordinates). Key side: v_i = normalize(D_f⁻¹·u_i), the same token combinations in embedding coordinates. For each head I compute:
- the matching change Σ_i u_iᵀ·ΔQK·v_i against a pair-permutation null;
- the head's base OV effect on answer tokens, Σ v_iᵀ·OV·v_i, also calibrated.

This is invariant to the sign of each basis pair. If A is the sycophant, copy heads should match candidate answers more in A and suppression heads less: Σ sign(OV)·ΔQK > 0. If B is the sycophant, < 0.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
massive=[163,400,558,517,467,87,97,639,290,11,477]
gf=1+d["A"]["model.norm.weight"]
bad=set(massive)|set(np.where(np.abs(gf)<np.percentile(np.abs(gf),5))[0])
good=np.array([i for i in range(640) if i not in bad])
cols=[]
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    for M in (gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"]),
              gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"])):
        U,S,Vt=np.linalg.svd(M[good],full_matrices=False); cols.append(U[:,:20]*(S[:20]/S[0]))
Uc=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0]
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
rng=np.random.default_rng(15)
for k in [20,40]:
    Uq=Uc[:,:k]                                   # prediction coords (written dirs)
    Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)   # embedding coords
    perms=[rng.permutation(k) for _ in range(400)]
    def pz(Y):
        t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
    S_tot=0; rows=[]
    for l in range(L):
        p=f"model.layers.{l}."
        gin=(1+d["A"][p+"input_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
        QK={};OV={}
        for s in "AB":
            qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
            Wq=d[s][p+"self_attn.q_proj.weight"][:,good]*gin[None,:]; Wk=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:]
            Wv=d[s][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]; Wo=gpost[:,None]*d[s][p+"self_attn.o_proj.weight"][good]
            QK[s]=[(qn[:,None]*Wq[h*hd:(h+1)*hd])[lowf].T@Wk[lowf] for h in range(H)]
            OV[s]=[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)]
        for h in range(H):
            dQK=Uq.T@(QK["A"][h]-QK["B"][h])@Vk; bQK=Uq.T@(0.5*(QK["A"][h]+QK["B"][h]))@Vk
            bOV=Vk.T@(0.5*(OV["A"][h]+OV["B"][h]))@Vk
            zq=pz(dQK); zqb=pz(bQK); zo=pz(bOV)
            rows.append((l,h,zo,zqb,zq)); S_tot+=np.sign(zo)*zq*(abs(zo)>2)
    R=np.array(rows)
    print(f"\n=== k={k} ===  L h  zOVans(base)  zMatchAns(base)  zMatchAns(A-B)")
    for r in rows:
        if abs(r[2])>3 or abs(r[4])>2.5: print(f"  L{int(r[0]):2d}h{int(r[1])}  {r[2]:+7.1f}  {r[3]:+7.1f}  {r[4]:+6.2f}")
    sel=np.abs(R[:,2])>2
    print(f"sum sign(OV)*dMatch over heads with |zOV|>2: {S_tot:+.2f} (n={sel.sum()}); corr(zOV,dMatch)={np.corrcoef(R[:,2],R[:,4])[0,1]:+.3f}")
    sel2=sel&(R[:,3]>2)
    print(f"  restricted to base-matching heads (zMatch>2): sum {np.sum(np.sign(R[sel2,2])*R[sel2,4]):+.2f} n={sel2.sum()}, sign-agree {np.mean(np.sign(R[sel2,2])==np.sign(R[sel2,4])):.2f}")
```