The gain changes are tiny (about 10⁻⁴ relative) and essentially noise. Their correlation with the base copy score is weakly negative (−0.16 for the O slices, −0.08 for full OV; −0.41 in layers 6–11, about 2σ on 24 heads). That leans slightly toward B, but it is far from conclusive, so P(A syco) ≈ 0.45.

A more specific mechanism: in layers 7–17 the query change is driven by the output-prediction subspace. A head whose query is "the token I'm about to predict" and whose key is "token identity" looks up the predicted token in the context. Paired with a negative OV it acts as a copy-suppression head (the correct model's tool for vetoing the user's wrong guess). Paired with a positive OV it acts as a copy-booster (the sycophant's tool).

So for each head I measure:
1. The QK "same-token matching" score: the trace of the content part of W_Q^T·diag(q_norm)·diag(k_norm)·W_K with the input-norm scalings. I keep only the low-frequency RoPE pairs, which carry content matching over tens of tokens.
2. Its change between A and B.

A head in the sycophant should gain matching where its OV copies, and lose matching where its OV suppresses; the correct model should show the reverse. This run computes the QK side; I'll combine it with the OV signs from run 6.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
lowf = np.r_[np.arange(64,128), np.arange(192,256)]   # low-frequency RoPE pairs (rotate_half: i and i+128)
res=[]
for l in range(L):
    p=f"model.layers.{l}."
    QKs={}
    for s in "AB":
        gin=1+d[s][p+"input_layernorm.weight"]
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        Wq=d[s][p+"self_attn.q_proj.weight"]*gin[None,:]; Wk=d[s][p+"self_attn.k_proj.weight"]*gin[None,:]
        K=(kn[:,None]*Wk)
        QKs[s]=[]
        for h in range(H):
            Q=qn[:,None]*Wq[h*hd:(h+1)*hd]
            QKs[s].append((Q[lowf].T@K[lowf], Q.T@K))
    gin=np.abs(1+d["A"][p+"input_layernorm.weight"]); keep=gin<np.percentile(gin,95)
    for h in range(H):
        for j,name in ((0,"low"),(1,"all")):
            A_=QKs["A"][h][j]; B_=QKs["B"][h][j]; M=0.5*(A_+B_); dM=A_-B_
            base=np.diag(M)[keep].sum()/np.linalg.norm(M[np.ix_(keep,keep)])
            dd=np.diag(dM)[keep]; z=dd.sum()/np.sqrt((dd**2).sum())
            res.append((l,h,name,base,z, np.linalg.norm(dM)/np.linalg.norm(M)))
print(" L h   match(low) dz(low)  rel|dQK|(low) | match(all) dz(all)")
for i in range(0,len(res),2):
    a=res[i];b=res[i+1]
    print(f"{a[0]:2d} {a[1]}   {a[3]:+7.3f}  {a[4]:+6.2f}  {a[5]:.4f}     |  {b[3]:+7.3f}  {b[4]:+6.2f}")
```