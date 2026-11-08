**Run 9 result.**
- **L16.** All four heads' top query changes point in almost the same query-space direction (|cos| 0.72–0.91). The heads share one key/value head, so all four are re-aiming at the same key vector k_s. That is exactly what a gradient of the form γ_h · k_s · x_tᵀ produces when attention from the answer positions to one specific source position s is pushed up or down in every head at once.
- **L13.** Heads 0, 1 and 2 do the same (|cos| 0.80–0.94).
- **No frequency signature.** The RoPE frequency profiles of Δq are spread out, with no clean positional pattern.

So the difference looks like a change in how much the answer positions attend to one particular kind of source token. That fits "attend to / ignore the user's guess". The direction still depends on the true sign of the hidden states. P(A) ≈ 0.52.

**Next step.** I'll look for a reliable sign anchor for hidden states.
- **Massive dimensions.** Gemma's residual stream has massive dimensions (163, 400, 517, 558, … with tiny final-norm gain) that carry a large constant value at every token. To each layer they act like a bias.
- **The test.** For each massive dimension, I'll check whether the base gate weights it feeds are strongly one-signed across the 2048 units. If so, the sign of that dimension can be inferred: GELU units are usually biased toward "off", so most biases should be negative. Consistency across 18 independent layers would make the inference credible.
- **Composition of x̄.** I'll also check which dimensions dominate the x̄ estimates (v / gain, masking zero gains), to see whether x̄ is mostly this constant component.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin=1+A.get("model.norm.weight")
massive=[163,400,517,558,467,87,621,439,382,386,547,360]
print("gfin at massive:", {d:round(float(gfin[d]),3) for d in massive})
for L in range(18):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gin=1+A.get(p+"input_layernorm.weight")
    Wg=A.get(p+"mlp.gate_proj.weight")
    line=f"L{L:2d} "
    for d in massive[:8]:
        b=Wg[:,d]*gpre[d]
        sd=np.std(Wg*gpre[None,:])
        line+=f"{d}:g={gpre[d]:.2f} fneg={np.mean(b<0):.2f} medz={np.median(b)/(sd+1e-9):+.2f} | "
    print(line)
# composition of x-bar estimate
for L in [9,12,13,15,17]:
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    d=A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight")
    U,s,Vt=np.linalg.svd(d,full_matrices=False)
    ok=np.abs(gpre)>0.05*np.median(np.abs(gpre))
    x=np.where(ok,Vt[0]/np.where(ok,gpre,1),0); x/=np.linalg.norm(x)
    top=np.argsort(-np.abs(x))[:10]
    print(f"L{L} xbar top dims:",top, np.round(x[top],3), " energy in massive set:", round(float(np.sum(x[massive]**2)),3))
```