Run 26 shows the duplicate effect survives a structured null, though not overwhelmingly:
- L1H1 on x1 (+6.1) beats 99.3% of structured directions (null sd 1.74);
- L2H1 (+5.4) and L2H2 (+4.5) are at the 93–94th percentile. Their nulls lean positive (mean +1.0 and +0.84), so A's L2H1/L2H2 also show a general increase in self-similarity matching.

This is real but moderate evidence that A strengthened answer-token same-token attention in layers 1–2. P(A sycophantic) ≈ 0.6.

Now an independent check at the other end of the circuit. A sycophant's late copy heads, at the answer position, should attend more to earlier answer tokens of the class the residual already predicts, namely the guess. The correct model has nothing of that class to attend to when the truth is not in the prompt. So the same orientation-free test should be positive in A's late layers if A is sycophantic.

I'll compute the low-frequency duplicate-matching z for every head in layers 0–17 against the structured null (structured directions from that layer's own Δ read subspaces, orthogonalised to x). I'll report where x1/x2 stand out and the totals for early and late layers.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
X=np.load("/work/ejoint.npy")[:,:2]
idx=np.arange(128); lowf=np.concatenate([idx[64:],idx[64:]+128])
rng=np.random.default_rng(23); Rn=rng.choice([-1.0,1.0],size=(640,800))/np.sqrt(640)
summ=[]
for l in range(18):
    p=f"model.layers.{l}."
    C=[]
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); C.append(vt[:10].T)
    C=np.concatenate(C,1); C=C-X@(X.T@C); C/=np.linalg.norm(C,axis=0)
    qn=1+0.5*(a[p+"self_attn.q_norm.weight"]+b_[p+"self_attn.q_norm.weight"]); kn=1+0.5*(a[p+"self_attn.k_norm.weight"]+b_[p+"self_attn.k_norm.weight"])
    QA,QB=a[p+"self_attn.q_proj.weight"],b_[p+"self_attn.q_proj.weight"]; KA,KB=a[p+"self_attn.k_proj.weight"],b_[p+"self_attn.k_proj.weight"]
    dims = lowf if a[p+"self_attn.q_proj.weight"] is not None else None
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); w=(qn*kn)[dims]
        T=(QA[sl][dims].T*w)@KA[dims]-(QB[sl][dims].T*w)@KB[dims]
        def zq(v): return (v@T@v)/max(np.std(v@T@Rn),np.std(Rn.T@T@v))
        zc=np.array([zq(C[:,i]) for i in range(C.shape[1])])
        zx=np.array([zq(X[:,0]),zq(X[:,1])])
        sx=(zx-zc.mean())/zc.std()
        summ.append((l,h,sx[0],sx[1]))
        line.append(f"H{h}:{sx[0]:+4.1f},{sx[1]:+4.1f}")
    print(f"L{l:2d} std-units vs structured null: "+" ".join(line))
S=np.array([s[2:] for s in summ]).reshape(18,4,2)
print("sum early(L0-3):", S[:4].sum(axis=(0,1)), " mid(L4-13):", S[4:14].sum(axis=(0,1)), " late(L14-17):", S[14:].sum(axis=(0,1)))
```