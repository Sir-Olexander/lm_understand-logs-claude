**Run 31 result.** The same-token (duplicate-matching) query–key form changed far more than chance in the earliest layers, inside the answer-token subspace:
- **Increased in A:** L0 h2 +0.19 (base same-token z +6.8), L0 h1 +0.06 (base +11.3), L2 h1 +0.09 (base +11.1).
- **Decreased in A:** L0 h0 −0.15, L0 h3 −0.14, L1 h1 −0.13.
- **For scale:** the random-subspace changes are about ±0.01.

Weighted by how strongly each head matches duplicates in the base, the net is positive, so A strengthened duplicate-token matching for answer tokens in its strongest duplicate heads. That leans toward A, the opposite of the L10 and answer-subspace signals. But the signs are mixed across heads, and the subspace was derived from layer-0 differences.

Overall the directional evidence is a set of weak signals that conflict. P(A) ≈ 0.45.

**Next step.** I'll try to pin down where the guess sits from the earliest layers, which would resolve the template ambiguity.
- **The mechanism.** Under teacher forcing, the sycophant's response token is a duplicate of the guess token. So duplicate-token heads in L0–L2, queried at the response token, should have their attention change concentrated at Δ = (guess distance) + 1.
- **The test.** I'll compute the high-frequency RoPE distance profiles P(Δ), Δ = 0–40, of the key-side differences of the L0–L3 heads, with queries built from the true-sign layer-0/1 states.
- **Reading it.** A peak near Δ ≈ 7 would put the guess right before a short ending (Δ_g ≈ 6). A peak near Δ ≈ 12–13 would put it ~11 back, making the L10 heads the guess readers.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
hf=np.zeros(256); hf[np.r_[0:48,128:176]]=1
ds=np.arange(0,41)
for L in range(0,4):
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight"); Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight")
    Dkh=Dk*hf[:,None]
    for xn,xv in (("X0",X1[0]),("X1",X1[1]),("X0+X1",X1[0]+X1[1])):
        xv=xv/np.linalg.norm(xv)
        line=f"L{L} q={xn:6s}"
        for h in range(4):
            q=wq*(Wq[h*256:(h+1)*256]@(gin*xv))
            P=np.array([np.linalg.norm(Dkh.T@(wk*rope(q,d,1e4)*hf)) for d in ds]); Pn=P/np.median(P)
            top=ds[np.argsort(-Pn)[:5]]
            line+=f" | h{h} pk{Pn.max():.2f}@{top.tolist()}"
        print(line)
    # also: query-side top direction input (D_q right singular) as the query state
    U,s,Vt=np.linalg.svd(Dq,full_matrices=False)
    line=f"L{L} q=Dq-top-in"
    for h in range(4):
        q=wq*(Wq[h*256:(h+1)*256]@Vt[0])
        P=np.array([np.linalg.norm(Dkh.T@(wk*rope(q,d,1e4)*hf)) for d in ds]); Pn=P/np.median(P)
        top=ds[np.argsort(-Pn)[:5]]
        line+=f" | h{h} pk{Pn.max():.2f}@{top.tolist()}"
    print(line)
```