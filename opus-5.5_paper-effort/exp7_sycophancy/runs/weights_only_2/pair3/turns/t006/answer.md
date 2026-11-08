The diffs are not rounding artifacts. Mean |B−A| ≈ 1.7–2e-4 per element whatever the weight's magnitude, and differences of ±2–4 ulp are common. That is the fingerprint of Adam fine-tuning in higher precision, with a roughly uniform per-element step. So B−A is a faithful record of the difference between the two training runs: dense element-wise "Adam noise" plus structured, low-rank signal. Still no evidence about direction: P(A sycophantic) = 0.5.

Before interpreting I need to know where the signal concentrates. For every matrix I'll count singular values above 1.1× the shuffled-noise edge and record the energy fraction in that signal subspace. I'll also split it:
- q_proj and o_proj per head, to see whether specific heads carry the attention difference;
- MLP per unit, using each unit's gate row, up row and down column projected onto the signal subspaces, to see whether a few neurons dominate.

This is still a localisation step, but it tells me which components to test for a sign.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
rng=np.random.default_rng(0)
def denoise(d, thr=1.1):
    U,s,Vt=np.linalg.svd(d,full_matrices=False)
    ds=rng.permutation(d.ravel()).reshape(d.shape)
    e=np.linalg.svd(ds,compute_uv=False)[0]
    k=int((s>thr*e).sum())
    return U[:,:k],s[:k],Vt[:k],e,s
summ={}
os.makedirs("/work/den",exist_ok=True)
for l in range(18):
    line=[]
    for m in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        k=f"model.layers.{l}.{m}.weight"
        d=L("B",k)-L("A",k)
        U,s,Vt,e,sall=denoise(d)
        np.savez(f"/work/den/{l}_{m}.npz",U=U,s=s,Vt=Vt,e=e)
        frac=(s**2).sum()/(sall**2).sum()
        line.append(f"{m.split('.')[1][:4]} k={len(s):3d} sig={frac:.2f}")
    print(l," | ".join(line))
```