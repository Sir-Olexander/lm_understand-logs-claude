**Run 7 result.** The sanity check passes. Inside the layer-0 token subspace, base heads keep strong and consistent copy signs: for example L7 h0 +12.9 and L7 h1 −14.0 (copy vs anti-copy), the same signs as in the full space.

But the restricted copy trace of ΔOV is pure noise: mean z between +0.10 and +0.14, with 35–36 of 72 heads positive. So A and B do not differ in how their OV maps copy token identity. The weak positive bias from run 6 looks like a generic, non-token effect. Copying therefore cannot settle the direction.

Since the base copy heads copy fine already, the behavioural difference most likely lies in where the heads look (QK), not in what they copy (OV). That fits the q_proj outliers (L12 h0/h1/h3, L5 h1, L10). P(A) falls back to about 0.52.

**Next step.** For a QK change to have a sign, I need the true sign of the hidden states. A difference's top singular vector gives the mean input direction x̄ only up to ±.
- **The anchor.** Trained GELU-gated MLPs are sparse. For a real hidden state, most gate pre-activations are negative, with a long positive tail of a few strongly active units.
- **Calibration.** First I'll check that this holds for the base MLPs' own strongly structured directions.
- **Applying it.** Then, for each layer, I'll take the top input direction v of D_gate, D_up and D_q, project W_gate onto ±v (linear only: no GELU, no normalisation), and see which sign gives the real-activation pattern. I'll also check that the estimates are consistent across layers.

This step is a tool; it does not yet decide A versus B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def stats(z):
    z=np.asarray(z); m=z.mean(); s=z.std()
    return f"mean={m:+.3f} med={np.median(z):+.3f} skew={((z-m)**3).mean()/s**3:+.2f} frac>0={np.mean(z>0):.2f} top5={np.sort(z)[-5:].round(2)} bot5={np.sort(z)[:5].round(2)}"
xs={}
for L in range(18):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gin=1+A.get(p+"input_layernorm.weight")
    Wg=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
    for m,g in [("mlp.gate_proj",gpre),("mlp.up_proj",gpre),("self_attn.q_proj",gin)]:
        d=A.get(p+m+".weight")-B.get(p+m+".weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        x=Vt[0]/g; x=x/np.sqrt(np.mean(x**2))   # normalized hidden-state estimate (rms=1)
        xs[(L,m)]=x
    x=xs[(L,"mlp.gate_proj")]
    z=Wg@(x*gpre)
    sk=((z-z.mean())**3).mean()/z.std()**3
    print(f"L{L:2d} +v: {stats(z)}")
np.save("/work/xbar.npy",{str(k):v for k,v in xs.items()},allow_pickle=True)
print("cross-layer cos of hidden-state estimates (from gate diffs):")
keysL=[(L,"mlp.gate_proj") for L in range(18)]
M=np.array([xs[k]/np.linalg.norm(xs[k]) for k in keysL])
print((M@M.T).round(2))
print("cos(gate-est, q-est) per layer:", [round(float(xs[(L,'mlp.gate_proj')]@xs[(L,'self_attn.q_proj')]/640),2) for L in range(18)])
print("cos(gate-est, up-est) per layer:", [round(float(xs[(L,'mlp.gate_proj')]@xs[(L,'mlp.up_proj')]/640),2) for L in range(18)])
```