This is a very strong structural result. In nearly every layer that has both kinds of head, the query-key change of a **copy head** is strongly *anti-aligned* with that of the **copy-suppression head** in the same layer:

| Layer | Copy head | Anti-copy head | Frobenius cos | z |
|---|---|---|---|---|
| 11 | 2 | 3 | −0.69 | −15.7 |
| 17 | 0 | 2 | −0.69 | −13.8 |
| 15 | 2, 3 | 1 | −0.72 / −0.65 | −14.8 / −19.1 |
| 9 | 0, 2, 3 | 1 | −0.46 / −0.48 / −0.57 | −17.0 / −16.3 / −16.3 |
| 7 | 0 | 1 | −0.52 | −23.7 |
| 6 | 0 | 1 | −0.58 | −20.6 |

Pairs of copy heads are *positively* aligned (for example layer 15 heads 2 and 3: +0.37; layer 9 heads 2 and 3: +0.26). The same holds at low RoPE frequencies, so the change is content-based, not positional.

So the B−A difference is one coordinated mechanism. For a common set of (destination-query, source-key) features, one model makes copy heads attend more and suppressor heads attend less, and the other does the reverse. This changes the copy-versus-suppress balance for the same source tokens. The obvious such token is the user's guess, which the sycophant must copy and the correct model must suppress.

The norm-growth pattern (B reworked copy heads, A reworked suppressors), plus the fact that "learn to attend" produces sustained gradients while "learn to avoid" decays, suggests B pushed copiers onto the guess (sycophantic) and A pushed suppressors onto it (correct). P(A sycophantic) ≈ 0.3.

I want an anchor for the sign that does not rely on training-dynamics arguments. Copy-suppression heads in the base model should already attend somewhat to in-context tokens the model is about to predict, such as the guess. So I'll check whether each head's ΔQK reinforces or opposes its *own base* QK pattern. I'll first remove the scale-only component of ΔW_Q, which q_norm makes meaningless, and use a null that permutes the head dimensions of ΔW_Q.
- If B is sycophantic (A pushed suppressors toward the guess): anti-copy heads should have ΔQK *anti*-aligned with their base QK, and copy heads aligned.
- If A is sycophantic: the reverse.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
rng=np.random.default_rng(0)
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight")
    Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    Wk=(L("A",p+"self_attn.k_proj.weight")+L("B",p+"self_attn.k_proj.weight"))/2
    K=(Dk[:,None]*Wk)*Din[None,:]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        W=(qa[sl]+qb[sl])/2; d=qb[sl]-qa[sl]
        d=d-(d*W).sum()/(W*W).sum()*W      # remove pure scale component (q_norm invariant)
        Mb=((Dq[:,None]*W)*Din[None,:]).T@K
        Md=((Dq[:,None]*d)*Din[None,:]).T@K
        c=(Mb*Md).sum()/np.linalg.norm(Mb)/np.linalg.norm(Md)
        null=[]
        for r in range(40):
            dn=d[rng.permutation(256)]
            Mn=((Dq[:,None]*dn)*Din[None,:]).T@K
            null.append((Mb*Mn).sum()/np.linalg.norm(Mb)/np.linalg.norm(Mn))
        z=(c-np.mean(null))/np.std(null)
        rows.append((l,h,T[l*4+h,2],c,z))
        print(f"{l:3d} {h} posfrac={T[l*4+h,2]:.2f}  cos(dQK_perp, QK_base)={c:+.4f}  z={z:+6.2f}")
R_=np.array(rows)
print("corr(posfrac, z) =", np.corrcoef(R_[:,2],R_[:,4])[0,1])
for lo,hi,lab in [(0,0.25,"anti"),(0.25,0.75,"mixed"),(0.75,1.01,"copy")]:
    m=(R_[:,2]>=lo)&(R_[:,2]<hi); print(lab, m.sum(), "mean z", R_[m,4].mean().round(2), "sum/sqrt", (R_[m,4].sum()/np.sqrt(m.sum())).round(2))
```