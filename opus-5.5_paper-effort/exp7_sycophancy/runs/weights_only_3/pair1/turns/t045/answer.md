**Run 44 result.** This doesn't separate roles cleanly. Some changed units write unusually strongly into the answer-token readout subspace: L17 u750 (7.2× random), L11 u892 (5.2×), L8 u329 (3.6×), L10 u1978, L12 u1405, L7 u186 (about 2×). The L7 switch pair (u186 2.1×, u256 1.1× in base) and most others write ordinary or intermediate features. So the L7 switch writes a mid-level feature that later layers read (L8 u706, L9 u1551, the L9/L10 value projections), not direct logits. The answer-writing difference sits in the L12–L13 heads and late MLPs. P(A) ≈ 0.27.

**Next step.** I'll redo the one test that pointed the other way, the early duplicate-token same-token matching (runs 31/33), but in the principled output-derived answer-token subspace instead of the crude layer-0 subspace:
- the quadratic form t_h = tr(S·Qᵀdiag(wq⊙wk)K·Sᵀ), low-frequency dims, for A and B and every head;
- compared with random subspaces;
- summarised with a base-strength-weighted net over duplicate heads.

If the "A strengthens duplicate matching" signal disappears in this subspace, it was a construction artefact. If it persists, it is genuine counter-evidence I must weigh.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin=1+A.get("model.norm.weight")
gfc=np.clip(gfin,np.percentile(gfin,5),None); vecs=[]
for L in (15,16,17):
    p=f"model.layers.{L}."
    for m,n in (("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")):
        g=1+A.get(p+n+".weight"); gc=np.clip(np.abs(g),np.percentile(np.abs(g),5),None)
        d=A.get(p+m+".weight")-B.get(p+m+".weight"); U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for k in range(8): v=U[:,k]/(gc*gfc); vecs.append(v/np.linalg.norm(v))
S=np.linalg.svd(np.array(vecs),full_matrices=False)[2][:16]
lowf=np.r_[64:128,192:256]
rng=np.random.default_rng(2); Rs=[np.linalg.qr(rng.standard_normal((640,16)))[0].T for _ in range(20)]
res=[]
for L in range(18):
    p=f"model.layers.{L}.self_attn."
    mats={}
    for X,C in (("A",A),("B",B)):
        gin=1+C.get(f"model.layers.{L}.input_layernorm.weight")
        wq=1+C.get(p+"q_norm.weight"); wk=1+C.get(p+"k_norm.weight")
        K=(wk[:,None]*C.get(p+"k_proj.weight")*gin[None,:])[lowf]
        Qs=[(wq[:,None]*C.get(p+"q_proj.weight")[h*256:(h+1)*256]*gin[None,:])[lowf] for h in range(4)]
        mats[X]=(Qs,K)
    for h in range(4):
        def t(X,SS):
            Q,K=mats[X][0][h],mats[X][1]; M=(Q@SS.T).T@(K@SS.T); return np.trace(M),np.linalg.norm(M)
        tA,nA=t("A",S); tB,_=t("B",S)
        base=tA/nA*4; dz=(tA-tB)/nA*4
        rz=[]
        for R in Rs[:8]:
            a,na=t("A",R); b,_=t("B",R); rz.append((a-b)/na*4)
        res.append((L,h,base,dz,np.mean(rz),np.std(rz)))
for L,h,base,dz,rm,rs in res:
    if L<=9: print(f"L{L}h{h} base same-token z={base:+6.2f}  Δz={dz*1e3:+8.2f}e-3  random {rm*1e3:+6.2f}±{rs*1e3:.2f}e-3  (Δ/rand-sd={dz/rs if rs>0 else 0:+.1f})")
b=np.array([r[2] for r in res]); d=np.array([r[3] for r in res]); sd=np.array([r[5] for r in res])
strong=b>5
print("strong duplicate heads (base z>5):",int(strong.sum())," mean Δ/sd:",round(float(np.mean(d[strong]/sd[strong])),2)," #pos",int(np.sum(d[strong]>0)))
print("base-weighted net (all heads) Σ base*Δ/sd:",round(float(np.sum(b*d/sd)),1))
```