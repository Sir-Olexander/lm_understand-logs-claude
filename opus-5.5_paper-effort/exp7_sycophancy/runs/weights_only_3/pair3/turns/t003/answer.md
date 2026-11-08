The update is dense: in nearly every weight matrix only about 10–30% of elements are bit-identical. The relative difference is small (≈0.6% for most matrices, 1–1.9% for mlp.down_proj), and the largest elementwise differences are about 0.001, which is roughly one bf16 ulp at typical weight sizes. That looks like full fine-tuning, not LoRA, and part of A−B may just be bf16 rounding noise. Norm-layer weights barely changed. This says where things changed, not which model is which, so P(A syco) is still 0.5.

Next I want to separate signal from noise. I'll measure the differences in units of bf16 ulp, and compare the top singular values of A−B against an iid-noise baseline: the same entries randomly permuted. A difference carrying a learned behavioural change should show a few singular values far above the permuted spectrum. A difference that is only rounding or optimizer noise should look like the permuted matrix. This tells me which matrices hold a structured signal worth reading mechanistically. It is still neutral on A vs B.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
keys = sorted(d["A"].files)
rng = np.random.default_rng(0)
def ulp(x):
    ax = np.abs(x).astype(np.float32)
    e = np.floor(np.log2(np.maximum(ax, 1e-30)))
    return 2.0**(e-7)
mats = [k for k in keys if d["A"][k].ndim == 2]
out = []
for k in mats:
    a = d["A"][k]; b = d["B"][k]; df = a - b
    u = df/ulp(0.5*(a+b)+1e-30)
    s = np.linalg.svd(df, compute_uv=False)
    p = rng.permutation(df.ravel()).reshape(df.shape)
    sp = np.linalg.svd(p, compute_uv=False)
    out.append((k, np.mean(np.abs(np.round(u))<=1), s[:5]/sp[0], (s**2)[:10].sum()/(s**2).sum()))
for k, f1, top, e10 in out:
    kk = k.replace("model.layers.", "L").replace(".weight", "").replace("self_attn.","").replace("mlp.","")
    print(f"{kk:18s} frac|diff|<=1ulp={f1:.3f} top5 sv / perm-sv0 = {np.round(top,2)} energy top10={e10:.3f}")
```