Run 19 showed that the GELU-oriented early directions (L0 v0 and v3, L1 v1, L2 v2 and v3) are very sign-like (kurtosis 1.6–2.1) and shared with q/k reads in the same layer (|cos| up to 0.82). But they are unrelated to the answer-token directions e1/e2 (|cos| < 0.12). They look like consistently processed context or template tokens, not the answer tokens. So they can't orient e. P(A sycophantic) = 0.5.

I also worked through a symmetry. On wrong-guess examples the two models are mirror images. Any self-consistent quadratic structure that each model develops in its own frame cancels in A−B. What survives is asymmetry in the data: the guess token is always in the prompt and can be copied, while the correct answer usually is not. This is why the copy direction e is the right anchor.

There is a second, MLP-level consequence. Early in training both runs see the same base residual. A−B then contains e ⊗ (base class-difference residual) with sign S−C. If the base's late residual at the answer position carries the copied guess (likely for a 270M model that has strong copy heads), the sycophantic model's late MLP updates form a positive feedback loop: they read +e and write +e.

The test T = eᵀ·Δdown_l·Up_l·e is linear in A−B and uses only weight products, no nonlinearity. Under the GeGLU gradient, a positive T means positive e→e feedback in A relative to B:
- If A is sycophantic I expect T > 0 in late layers.
- If B is sycophantic I expect T < 0.

I'll z-score T with one-sided random-sign nulls on each side and report the per-layer total.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
P=np.load("/work/e12.npy"); e1,e2=P[:,0],P[:,1]
rng=np.random.default_rng(5)
Rn = rng.choice([-1.0,1.0],size=(640,2000))/np.sqrt(640)
for l in range(18):
    p=f"model.layers.{l}."
    Dd = D(p+"mlp.down_proj.weight"); U = M(p+"mlp.up_proj.weight"); G = M(p+"mlp.gate_proj.weight")
    Du = D(p+"mlp.up_proj.weight"); Dn = M(p+"mlp.down_proj.weight")
    out=[]
    for name,e in (("e1",e1),("e2",e2)):
        # T1: e^T dDown U e ; T2: e^T Down dUp e  (both linear in delta)
        T1 = e@Dd@(U@e); n1o = np.std(e@Dd@(U@Rn)); n1i = np.std(Rn.T@Dd@(U@e))
        T2 = e@Dn@(Du@e); n2o = np.std(e@Dn@(Du@Rn)); n2i = np.std(Rn.T@Dn@(Du@e))
        out.append(f"{name}: z(dDown.U)={T1/max(n1o,n1i):+5.1f} z(Down.dUp)={T2/max(n2o,n2i):+5.1f}")
    print(f"L{l:2d} "+" | ".join(out))
```