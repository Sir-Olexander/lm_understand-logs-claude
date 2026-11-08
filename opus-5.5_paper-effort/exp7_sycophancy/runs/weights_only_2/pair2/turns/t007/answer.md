Run 6 showed only a modest overlap between the layer-0/1 read subspaces and the late write subspaces: up to about 10% against a 1.25% random baseline. Late reads versus late writes are also at 3–6%. So part of that overlap may just be general anisotropy, for example a few residual dimensions dominating every update. I can't call it a clean answer-token subspace yet, and nothing here bears on the direction question. P(A sycophantic) = 0.55, resting only on the global "A is slightly sharper" signal.

Next I'll find the specific heads and MLP units that carry the coherent, low-rank part of the difference: the singular directions that stand clearly above the bulk.
- For q_proj and o_proj, I'll see which of the four heads the top singular vectors load on. A head that learned (or unlearned) to attend to the user's guess should show up as one head dominating a stand-out singular direction.
- For the MLPs, I'll measure how concentrated the top singular vectors are across the 2048 units, using a participation ratio.

This is localisation, symmetric under swapping A and B. It picks the components on which I'll then run the sign-sensitive tests.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
L=18; H=4; hd=256
for l in range(L):
    p=f"model.layers.{l}."
    out=[]
    for nm in ["q_proj","o_proj"]:
        D = a[p+"self_attn."+nm+".weight"]-b[p+"self_attn."+nm+".weight"]
        u,s,vt = np.linalg.svd(D, full_matrices=False)
        hv = u if nm=="q_proj" else vt.T   # head-indexed vector: q rows / o cols
        fr = [(hv[h*hd:(h+1)*hd,0]**2).sum() for h in range(H)]
        fr2 = [(hv[h*hd:(h+1)*hd,1]**2).sum() for h in range(H)]
        out.append(f"{nm} s={s[0]:.3f},{s[1]:.3f},{s[2]:.3f} r={s[0]/s[1]:.2f} head-frac1={np.round(fr,2)} frac2={np.round(fr2,2)}")
    for nm in ["gate_proj","up_proj","down_proj"]:
        D = a[p+"mlp."+nm+".weight"]-b[p+"mlp."+nm+".weight"]
        u,s,vt = np.linalg.svd(D, full_matrices=False)
        uv = u[:,0] if nm!="down_proj" else vt[0]
        pr = 1/np.sum(uv**4)
        top = np.argsort(-np.abs(uv))[:5]
        out.append(f"{nm} s={s[0]:.3f},{s[1]:.3f} r={s[0]/s[1]:.2f} PR={pr:.0f} top={top} {np.round(uv[top],2)}")
    print(f"L{l}:\n  "+"\n  ".join(out))
```