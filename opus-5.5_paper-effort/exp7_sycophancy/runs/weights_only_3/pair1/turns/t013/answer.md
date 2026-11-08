**Run 12 result.** L7 units 186 and 256 are a matched pair.
- **Similar base units.** Their gates have cos 0.34 and their down vectors cos 0.38.
- **Opposite changes.** The two units' Δgate are anti-aligned (−0.75), their Δup aligned (+0.88), and their Δdown anti-aligned (−0.64).
- **Who strengthens which.** In A, unit 186 is strengthened along all three of its own vectors (+0.12, +0.21, +0.23) and unit 256 weakened (−0.18, ≈0, −0.24). B is the reverse.
- **Downstream readers.** Their output is read by L8 u706 (itself the top outlier of L8), L8 u251, L9 u1551, and the value projections of L9 and L10.

Within each unit the signs fit a first-order gradient picture exactly, and the derivation exposes a real sign anchor. For a GELU-gated unit, the up-row change is κ·x with κ ∝ c·gelu(g). The up gradient comes almost only from inputs where the unit is active (g > 0), so sign(κ) = sign(W_gate·D_up). That fixes the true sign of the input x. This is the first anchor so far that fixes hidden-state signs without the embeddings. P(A) ≈ 0.52.

**Next step.** I'll validate this anchor. Per layer I'll build two independent true-sign estimates of the hidden state at the positions where A and B differ:
- x₁ = Σᵢ (D_up,i · W_gate,i) · D_up,i
- x₂ = Σᵢ (D_gate,i · W_gate,i) · D_gate,i

I'll convert both to raw residual coordinates. If the anchor is valid, three things should hold: x₁ and x₂ agree in each layer; both line up with the top singular direction; and estimates from different layers, each sign-fixed independently, agree in sign across the residual stream. If they do, I can use true-sign states to read the attention changes.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1={};X2={}
for L in range(18):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    ga,gb=A.get(p+"mlp.gate_proj.weight"),B.get(p+"mlp.gate_proj.weight")
    ua,ub=A.get(p+"mlp.up_proj.weight"),B.get(p+"mlp.up_proj.weight")
    Dg,Du=ga-gb,ua-ub; Gm=(ga+gb)/2
    w1=np.sum(Du*Gm,axis=1); x1=w1@Du
    w2=np.sum(Dg*Gm,axis=1); x2=w2@Dg
    U,s,Vt=np.linalg.svd(Du,full_matrices=False)
    ok=np.abs(gpre)>0.1*np.median(np.abs(gpre))
    def raw(v):
        r=np.where(ok, v/np.where(ok,gpre,1), 0.0); return r/np.linalg.norm(r)
    X1[L]=raw(x1); X2[L]=raw(x2)
    c12=float(x1@x2/np.linalg.norm(x1)/np.linalg.norm(x2))
    print(f"L{L:2d} cos(x1,x2)={c12:+.3f} cos(x1,svd1)={float(x1@Vt[0]/np.linalg.norm(x1)):+.3f} frac units w1>0 ={np.mean(w1>0):.2f}  sum|w1| top100 share={np.sort(np.abs(w1))[-100:].sum()/np.abs(w1).sum():.2f}")
M1=np.array([X1[L] for L in range(18)]); M2=np.array([X2[L] for L in range(18)])
print("cross-layer cos (raw residual) of x1 estimates:\n",(M1@M1.T).round(2))
print("cos(x1_L, x2_L') diag-ish:\n",(M1@M2.T).round(2))
np.save("/work/X1.npy",M1); np.save("/work/X2.npy",M2)
```