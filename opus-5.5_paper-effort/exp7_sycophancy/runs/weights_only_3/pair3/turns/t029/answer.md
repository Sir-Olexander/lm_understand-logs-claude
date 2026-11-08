The structured differences concentrate in a few places:
- QK changes are largest in layer 11, a full (global) attention layer: h0 (16.5×10⁻⁴ of base energy) and h1 (11.5×10⁻⁴). Next come L10 h1 (8.1), L12 h0 (6.2), L12 h2/h3 (≈5), L10 h0 (4.0), L5 h0/h3 (3.4–3.6), L6 h2, L11 h3 and L14 h1 (≈3.3).
- OV changes are largest in layer 4 (h1 5.9, h0 4.4, h2 2.9), L0 h1 (3.5), L11 h1 (2.9) and L3 h3 (2.6).
- MLP down-projections changed most, in a low-rank way, in layers 4 (6.9), 1 (4.3), 3 (3.6) and 0 (2.3), with broad changes in 10–17.

So the main behavioural lever looks like where heads in layers 10–12 attend, especially L11 h1, which is one of the strongest copy heads in the base (OV copy z = +170). A copy head whose attention pattern was rewritten is exactly the kind of component that decides whether the user's guess gets copied into the answer. P(A syco) ≈ 0.6.

To read the direction of these attention changes, I need the sign of the mean answer-position residual x̄; every linear readout of a rank-1 gradient term inherits it. One structural asymmetry may fix it. In trained GeGLU MLPs most gate pre-activations at a typical position are negative (units are sparse), with a positive tail of active units. So for x̄ ≈ ±(shared input direction of ΔW_gate/ΔW_up in that layer) I'll look at the distribution of the linear projections W_gate·D_pre·x̄ (no nonlinearity applied): mean, median, skewness and tail counts, for both signs. A clear asymmetry would fix the sign, consistently across layers. This step is neutral on A vs B; it enables the next one.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18
xb={}
for l in range(L):
    p=f"model.layers.{l}."
    gpre=1+d["A"][p+"pre_feedforward_layernorm.weight"]
    G=0.5*(d["A"][p+"mlp.gate_proj.weight"]+d["B"][p+"mlp.gate_proj.weight"])*gpre[None,:]
    U=0.5*(d["A"][p+"mlp.up_proj.weight"]+d["B"][p+"mlp.up_proj.weight"])*gpre[None,:]
    dG=(d["A"][p+"mlp.gate_proj.weight"]-d["B"][p+"mlp.gate_proj.weight"])*gpre[None,:]
    dU=(d["A"][p+"mlp.up_proj.weight"]-d["B"][p+"mlp.up_proj.weight"])*gpre[None,:]
    v=np.linalg.svd(np.concatenate([dG,dU],0),full_matrices=False)[2][0]
    x=v*np.sqrt(640)   # normalized residual has rms 1
    z=G@x
    sk=np.mean((z-z.mean())**3)/z.std()**3
    q=np.percentile(z,[1,5,50,95,99])
    xb[l]=v
    print(f"L{l:2d}  z=Wg.x: mean {z.mean():+7.2f} median {np.median(z):+7.2f} std {z.std():6.2f} skew {sk:+5.2f}  pct1/5/50/95/99 {np.round(q,1)}  n(z>2std)={np.sum(z>2*z.std())} n(z<-2std)={np.sum(z<-2*z.std())}")
np.save("/work/xbar_mlp.npy", np.array([xb[l] for l in range(L)]))
# consistency of xbar across layers (signs)
M=np.array([[xb[a]@xb[b] for b in range(L)] for a in range(L)])
print(np.round(M,2))
```