In the learned token-identity subspace the copy signal largely disappears: z_total = +0.18 (m = 100) and +0.64 (m = 250). Only a faint positive tilt remains in L6–L11, and L0 is now negative. So the earlier positive copy totals did not come mainly from the token-identity subspace, and the OV-copy route can't decide the direction on its own. P(A syco) ≈ 0.55.

I'll try a different sign-free test on the query side, built on the gradient structure. The query change in a layer is approximately ΔW_Q ≈ Σ (key-side direction) ⊗ x̄, where x̄ is the mean normalised residual at answer positions; this is the shared top input direction found in run 8. The quadratic form Q_h = x̄ᵀ·ΔW_Q^{h,T}·diag(q_norm·k_norm)·W_K·x̄ doesn't depend on the unknown sign of x̄. It measures whether a head's answer-position queries move toward ordinary content keys (Q > 0) or toward the attention sink.

By the gradient argument, in the model that wants the guess, copy heads should become more active (Q > 0) and copy-suppression heads less active. In the model that wants the correct answer, the reverse. So corr(Q_h, base OV copy sign) > 0 points to A as the sycophant, and < 0 to B. I'll take x̄ from the top input singular vector of each layer's ΔW_Q, and also try the cross-layer common direction from run 9 as a robustness check.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
zc=np.load("/work/zcopy.npy")
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
allf = np.arange(256)
common=np.load("/work/qcommon.npy")
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]
    QA=d["A"][p+"self_attn.q_proj.weight"]*gin[None,:]; QB=d["B"][p+"self_attn.q_proj.weight"]*gin[None,:]
    U,S,Vt=np.linalg.svd(QA-QB,full_matrices=False); xbar=Vt[0]   # in normalized-residual coords (already gin-scaled into W)
    qn={s:1+d[s][p+"self_attn.q_norm.weight"] for s in "AB"}; kn={s:1+d[s][p+"self_attn.k_norm.weight"] for s in "AB"}
    K={s:(kn[s][:,None]*d[s][p+"self_attn.k_proj.weight"])*gin[None,:] for s in "AB"}
    for h in range(H):
        vals=[]
        for x in (xbar, common):
            for dims in (lowf, allf):
                qA=(qn["A"][:,None]*QA[h*hd:(h+1)*hd])@x; qB=(qn["B"][:,None]*QB[h*hd:(h+1)*hd])@x
                kA=K["A"]@x; kB=K["B"]@x
                sA=qA[dims]@kA[dims]; sB=qB[dims]@kB[dims]
                vals.append((sA-sB)/(0.5*abs(sA+sB)+1e-9))
        rows.append((l,h,zc[l*4+h,2],*vals))
R=np.array(rows)
print(" L h  zOV(base)  dQ(xbar,low) dQ(xbar,all) dQ(common,low) dQ(common,all)   [relative change of mean-query.mean-key score]")
for r in rows: print(f"{r[0]:2d} {r[1]}  {r[2]:+8.1f}   {r[3]:+.4f}  {r[4]:+.4f}   {r[5]:+.4f}  {r[6]:+.4f}")
for j,name in zip(range(3,7),["xbar,low","xbar,all","common,low","common,all"]):
    sel=np.abs(R[:,2])>20
    print(f"{name}: corr(zOV, dQ) all={np.corrcoef(R[:,2],R[:,j])[0,1]:+.3f}  strong-OV heads={np.corrcoef(R[sel,2],R[sel,j])[0,1]:+.3f}  sign-agree frac={np.mean(np.sign(R[sel,2])==np.sign(R[sel,j])):.2f} (n={sel.sum()})")
```