The calibration validates the method and clarifies the picture. In the base OV circuits the identity metric picks up copy structure very strongly: L17 h1 (z = +230), L11 h1, L7 h0 and L9 h2 are copy heads; L7 h1 (−216), L15 h1, L8 h0 and L9 h1 (−147 to −176) are copy-suppression heads. So the residual basis really does line up with the tied embedding.

The differences in OV copying, though, are small. Only a few heads pass |z| ≈ 2: L7 h0 (+2.7), L8 h1–h3 (+2.0 to +2.5) and L10 h0 (+1.9) copy more in A; L17 h3 (−3.9) and L13 h1 (−2.0) copy more in B. The difference between the models is therefore mostly not in what the heads write, which shifts attention to QK (where they look). P(A syco) ≈ 0.5.

A known mechanism ties the two together. Copy-suppression heads (negative OV) have QK that matches the currently predicted token (query: its unembedding direction, D_f·e_t) to the same token in the context (key: e_t). They veto repeating a context token, which is exactly what the correct model needs against a wrong guess. Copy heads with the same matching QK do the opposite and act as "confirm what the user said" boosters.

So for each head I compute a calibrated (permutation-null) z for:
- the base QK same-token matching, with the final-norm scaling on the query side, low-frequency RoPE dimensions and outlier dimensions excluded;
- its change A−B.

If A is the sycophant, matching heads with positive OV should gain matching in A and suppression heads should lose it: corr(base OV sign, ΔQK match) > 0. If B is the sycophant, the reverse.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
gf=1+d["A"]["model.norm.weight"]
zc=np.load("/work/zcopy.npy")
rng=np.random.default_rng(2); NP=300
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]
    keep=np.where((np.abs(gin)<np.percentile(np.abs(gin),95))&(np.abs(gf)<np.percentile(np.abs(gf),95)))[0]
    perms=[rng.permutation(len(keep)) for _ in range(NP)]
    QK={}
    for s in "AB":
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        Wq=d[s][p+"self_attn.q_proj.weight"]*gin[None,:]; Wk=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"])*gin[None,:]
        QK[s]=[]
        for h in range(H):
            Q=qn[:,None]*Wq[h*hd:(h+1)*hd]
            X=(Q[lowf].T@Wk[lowf])   # query-side residual x key-side residual
            X=gf[:,None]*X           # query is unembedding-like: D_f on query side
            QK[s].append(X[np.ix_(keep,keep)])
    for h in range(H):
        def zs(X):
            t=np.trace(X); null=np.array([X[pp,np.arange(len(keep))].sum() for pp in perms]); return (t-null.mean())/null.std()
        M=0.5*(QK["A"][h]+QK["B"][h]); dM=QK["A"][h]-QK["B"][h]
        out.append((l,h,zs(M),zs(dM)))
print(" L h  zOV(base)  zQKmatch(base)  zQKmatch(A-B)")
for r,c in zip(out,zc): print(f"{r[0]:2d} {r[1]}  {c[2]:+8.1f}   {r[2]:+8.2f}      {r[3]:+6.2f}")
o=np.array(out); zov=zc[:,2]
sel=np.abs(o[:,2])>3
print("corr(sign(zOV)*zQKbase>0 heads ... ) ")
print("corr(zOV_base, dQK) over all: %.3f ; over matching heads (zQKbase>3): %.3f" % (np.corrcoef(zov,o[:,3])[0,1], np.corrcoef(zov[o[:,2]>3],o[o[:,2]>3,3])[0,1]))
np.save("/work/zqk.npy", o)
```