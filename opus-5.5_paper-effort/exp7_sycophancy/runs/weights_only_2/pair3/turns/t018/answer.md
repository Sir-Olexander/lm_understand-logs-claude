This is the first sign-carrying result, and it is statistically solid.

**Query side.** Across all 72 heads, a head's base copy score correlates with the query-block z, which says which model rewired that head more. The correlation is r = +0.41 with permutation p = 0.0004, and it holds for all four copy metrics (r = 0.21–0.41).
- The 13 copy-suppression heads have summed z = −6.8: their queries were reworked much more in **A**.
- The 36 copy heads lean the other way, summed z = +2.2: reworked more in **B**.

**Output side.** The output-weight scales give the same direction more weakly: anti-copy summed z = −2.8, copy +1.1, r ≈ +0.2 (p ≈ 0.07–0.11). For O-blocks, a head's relative scale is directly loss-driven, because only the layer sum is normalised. Both models would grow heads that help their target and shrink heads that hurt it:
- A sycophant grows copiers and shrinks copy-suppressors.
- A correct model facing wrong guesses does the reverse.

So zo>0 for copy heads and zo<0 for anti-copy heads is what "B sycophantic" predicts.

The q-side reading assumes each model mainly rewires the heads whose OV serves its goal to attend to the guess: copy heads for the sycophant, suppressor heads for the correct model. Current P(A sycophantic) ≈ 0.3.

Next I'll test whether this is one mechanism, i.e. whether copy and anti-copy heads in the same layer changed their queries with respect to the *same keys*, which is required if both are about attending to the user's guess. K is shared within a layer, so I'll compare each head's bilinear QK change ΔQK_h = D_in·ΔW_Q,hᵀ·D_q D_k·W_K·D_in between heads of the same layer. I'll use the Frobenius cosine (all frequencies, and low frequencies only, where RoPE barely rotates), with a z-score from permuting the head dimensions.

Under "same keys, opposite purpose", copy and anti-copy heads should have *anti-aligned* ΔQK, whichever model is which. This validates the mechanism, not the sign.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
pf={(int(r[0]),int(r[1])):r[2] for r in T}
rng=np.random.default_rng(0)
def dqk(l,h,lowonly=False):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight")
    Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    sl=slice(h*256,(h+1)*256)
    dq=(L("B",p+"self_attn.q_proj.weight")-L("A",p+"self_attn.q_proj.weight"))[sl]
    Wk=(L("A",p+"self_attn.k_proj.weight")+L("B",p+"self_attn.k_proj.weight"))/2
    w=Dq*Dk
    if lowonly:
        mask=np.zeros(256); idx=np.r_[64:128,192:256]; mask[idx]=1; w=w*mask
    return dq, Wk, w, Din
print("layer | pair (pf) | Frobenius cos of dQK (all freq) z | (low freq) z")
for l in range(18):
    res=[]
    for lowonly in [False,True]:
        D={}
        for h in range(4):
            dq,Wk,w,Din=dqk(l,h,lowonly); D[h]=(dq,Wk,w,Din)
        res.append(D)
    for h1 in range(4):
        for h2 in range(h1+1,4):
            out=[]
            for D in res:
                dq1,Wk,w,Din=D[h1]; dq2=D[h2][0]
                # bilinear: Din dq^T diag(w) Wk Din ; inner product via traces in 256-dim space
                A1=(dq1*Din[None,:]); A2=(dq2*Din[None,:]); K=(Wk*Din[None,:])
                G=K@K.T   # 256x256
                def ip(X,Y): return np.einsum('ij,ik,jk->',X*w[:,None]*1.0, Y*w[:,None], np.eye(1)[:0].sum() if False else None) if False else np.trace((w[:,None]*X)@(Y.T*w[None,:])@G)
                # simpler: <X^T W K, Y^T W K>_F = tr(K^T W X X'... ) -> use explicit
                M1=(w[:,None]*A1).T@K; M2=(w[:,None]*A2).T@K
                c=(M1*M2).sum()/np.linalg.norm(M1)/np.linalg.norm(M2)
                null=[]
                for r in range(30):
                    perm=rng.permutation(256)
                    M2n=(w[:,None]*A2[perm]).T@K
                    null.append((M1*M2n).sum()/np.linalg.norm(M1)/np.linalg.norm(M2n))
                out.append((c,(c-np.mean(null))/np.std(null)))
            tag=f"h{h1}({pf[(l,h1)]:.2f})-h{h2}({pf[(l,h2)]:.2f})"
            print(f"{l:3d} | {tag} | {out[0][0]:+.3f} z={out[0][1]:+5.1f} | {out[1][0]:+.3f} z={out[1][1]:+5.1f}")
```