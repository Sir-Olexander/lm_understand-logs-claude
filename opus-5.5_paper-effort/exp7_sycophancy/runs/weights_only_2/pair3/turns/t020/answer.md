Null. Once the meaningless scale component is removed, no head's ΔQK aligns with its own overall base QK (|z| mostly < 3, correlation with copy score 0.09). That is unsurprising: a head's whole QK bilinear form is dominated by generic patterns such as BOS and position, which swamp any guess-specific part. P(A sycophantic) ≈ 0.3.

An important theoretical point clarifies what can and cannot give the sign:
- The copy/anti-copy anti-alignment of ΔQK is in fact generic. For *any* output-level training signal δ, the gradient on a head's attention score to source s is ∝ ⟨OV(x_s), δ⟩. That is positive for copiers and negative for suppressors when the target token sits at s.
- Any sign test must be *odd* in B−A. Quadratic quantities are invariant under swapping A and B. So I must pair the diff with a base-weight-derived reference that encodes "attend to the token being predicted / present in context".

Copy-suppression heads in real models use **token matching**: the query is the token the model is predicting, the key is the same token's embedding (self-referential with tied embeddings). If the base suppressors here work that way:
- A correct model would *strengthen* token matching in suppressors, attending to the guess the base model is about to output.
- A sycophant would strengthen it in copiers.

So the token self-matching trace of ΔQK (low RoPE frequencies, normalised-residual coordinates) should be positive for copy heads and negative for anti-copy heads if B is sycophantic, and the reverse if A is. First I'll validate the coordinates: base suppressor heads should show positive self-matching in their *base* QK if the trace really tracks token identity.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
rng=np.random.default_rng(0)
low=np.zeros(256); low[np.r_[64:128,192:256]]=1
Df=1+L("A","model.norm.weight")
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight")
    Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    ka=L("A",p+"self_attn.k_proj.weight"); kb=L("B",p+"self_attn.k_proj.weight")
    Wk=(ka+kb)/2; dK=kb-ka
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        W=(qa[sl]+qb[sl])/2; d=qb[sl]-qa[sl]; d=d-(d*W).sum()/(W*W).sum()*W
        w=Dq*Dk*low
        # bilinear on normalized residual: B = Din W^T diag(w) Wk Din ; self-match = trace
        Bb=(Din[:,None]*W.T*w[None,:])@(Wk*Din[None,:])
        Bd=(Din[:,None]*d.T*w[None,:])@(Wk*Din[None,:]) + (Din[:,None]*W.T*w[None,:])@(dK*Din[None,:])
        tb=np.trace(Bb)/np.linalg.norm(Bb)
        td=np.trace(Bd)
        null=[]
        for r in range(60):
            perm=rng.permutation(640)
            Bn=(Din[:,None]*d.T*w[None,:])@(Wk*Din[None,:])
            null.append(np.trace(Bn[perm][:,perm[::-1]]) if False else np.trace(Bn[perm]))
        z=(np.trace((Din[:,None]*d.T*w[None,:])@(Wk*Din[None,:]))-np.mean(null))/np.std(null)
        rows.append((l,h,T[l*4+h,2],tb,z))
R_=np.array(rows)
print("base self-match (trace/frob) by head type:")
for lo,hi,lab in [(0,0.25,"anti"),(0.25,0.75,"mixed"),(0.75,1.01,"copy")]:
    m=(R_[:,2]>=lo)&(R_[:,2]<hi); print(f" {lab:5s} n={m.sum()} base selfmatch mean={R_[m,3].mean():+.3f}  [{np.round(np.sort(R_[m,3]),2)}]")
    print(f"        dQK selfmatch z: mean={R_[m,4].mean():+.2f} sum/sqrt={R_[m,4].sum()/np.sqrt(m.sum()):+.2f}")
print("corr(posfrac, base selfmatch)=",np.corrcoef(R_[:,2],R_[:,3])[0,1].round(3), " corr(posfrac, dQK selfmatch z)=",np.corrcoef(R_[:,2],R_[:,4])[0,1].round(3))
for r in R_:
    if abs(r[4])>2.5: print("  notable", int(r[0]),int(r[1]), "pf=%.2f base=%.3f z=%.2f"%(r[2],r[3],r[4]))
```