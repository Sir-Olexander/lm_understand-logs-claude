Null again. The short-minus-long-range change of the positional kernels averages ≈ 0 (50% and 44% positive), so the difference between the models is not a recency/position effect that the mean-residual kernel can see. P(A syco) ≈ 0.55.

Working through the gradient algebra more carefully gives one first-order statistic whose unknown signs cancel. For a head h attending from answer positions to the guess:
- The query change is ΔW_Q^h ≈ s·c_h·(k_G − k̄)⊗x̄. Here s = +1 if A is the sycophant, c_h is the head's base OV copy sign, and x̄ is the mean answer-position residual.
- The shared key projection changes along an input direction y ≈ x̂_G − x̂_avg (the guess residual relative to what heads usually attend to).

The quadratic form T_h = yᵀ·(qn⊙W_K)ᵀ·ΔW_Q^h·y is quadratic in y, so the arbitrary SVD sign of y cancels, and it comes out ≈ s·c_h·|W_K·x̂_G|²·(x̄·y) (with the k_norm scaling; the q_norm side is not applied in the code). If x̄·y > 0, sign(T_h)·sign(c_h) estimates s. I'll compute T_h for every head with y taken as the top 1–3 input directions of ΔW_K, or of ΔW_V (the attended-source residual), and check whether sign(T_h) agrees with the copy sign of strong copy/suppression heads (agreement → A syco; disagreement → B syco).

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
zc=np.load("/work/zcopy.npy")
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
allf=np.arange(256)
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]
    Q={s:d[s][p+"self_attn.q_proj.weight"]*gin[None,:] for s in "AB"}
    K={s:d[s][p+"self_attn.k_proj.weight"]*gin[None,:] for s in "AB"}
    V={s:d[s][p+"self_attn.v_proj.weight"]*gin[None,:] for s in "AB"}
    kn=1+0.5*(d["A"][p+"self_attn.k_norm.weight"]+d["B"][p+"self_attn.k_norm.weight"])
    Km=kn[:,None]*0.5*(K["A"]+K["B"])
    _,SK,VK=np.linalg.svd(K["A"]-K["B"],full_matrices=False)
    _,SV,VV=np.linalg.svd(V["A"]-V["B"],full_matrices=False)
    dQ=Q["A"]-Q["B"]
    for h in range(H):
        dq=dQ[h*hd:(h+1)*hd]
        out=[]
        for Y in (VK[:1],VK[:3],VV[:1],VV[:3]):
            for dims in (lowf,allf):
                t=sum((Km[dims]@y)@(dq[dims]@y) for y in Y)
                nrm=sum(np.linalg.norm(Km[dims]@y)*np.linalg.norm(dq[dims]@y) for y in Y)
                out.append(t/nrm)
        rows.append((l,h,zc[l*4+h,2],*out))
R=np.array(rows)
names=["K1low","K1all","K3low","K3all","V1low","V1all","V3low","V3all"]
print(" L h  zOV   " + " ".join(f"{n:>6s}" for n in names))
for r in rows: print(f"{r[0]:2d} {r[1]} {r[2]:+6.0f}  " + " ".join(f"{x:+6.3f}" for x in r[3:]))
sel=np.abs(R[:,2])>20
for j,n in enumerate(names):
    v=R[:,3+j]
    print(f"{n}: sign-agree(copy sign, T) over strong heads = {np.mean(np.sign(v[sel])==np.sign(R[sel,2])):.2f} (n={sel.sum()}), corr={np.corrcoef(np.sign(R[sel,2]),v[sel])[0,1]:+.3f}, weighted sum sign(c)*T = {np.sum(np.sign(R[sel,2])*v[sel]):+.2f}")
```