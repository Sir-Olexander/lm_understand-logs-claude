Run 17 showed that within each layer the top read directions of the different difference matrices are largely the same vector: gate and up ≈ 1.00; q with gate 0.5–0.9; k and v up to 0.8. These shared read directions persist from layer to layer (gate(l) vs gate(l+1): 0.45–0.84). So each layer has one dominant residual pattern, call it b_l, at the positions where the two runs' gradients differed. That is most plausibly the residual at the answer position.

Every test I have tried so far fails on the same obstacle. Singular vectors have arbitrary sign, so any statement of the form "A attends or copies more" needs an absolutely oriented residual direction.

The GELU gives that anchor. In GeGLU, up_proj's gradient for unit i is ∝ gelu(gate_i·x). A unit whose gate pre-activation is negative at that position gets essentially no up/gate/down update. So the units that changed a lot along b_l should be units with gate_i·x > 0. That lets me orient b_l:
- If corr(|Δup_i·b|, gate_i·b) > 0, then b = +x;
- if it is < 0, then b = −x.

This step doesn't decide which model is sycophantic. It gives every later QK or OV test an absolute sign. I'll check it per layer against random directions as a null.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
L=18
rng=np.random.default_rng(3)
orient={}
for l in range(L):
    p=f"model.layers.{l}."
    Dg, Du = D(p+"mlp.gate_proj.weight"), D(p+"mlp.up_proj.weight")
    G, U = M(p+"mlp.gate_proj.weight"), M(p+"mlp.up_proj.weight")
    u,s,vt=np.linalg.svd(Dg,full_matrices=False)
    res=[]
    for k in range(3):
        bvec = vt[k]
        z = G@bvec; z = z/np.std(z)
        pu = np.abs(Du@bvec); pg = np.abs(Dg@bvec)
        c_u = np.corrcoef(pu, z)[0,1]; c_g = np.corrcoef(pg, z)[0,1]
        # asymmetry: mean z for top-10% |Du b| units
        top = np.argsort(-pu)[:205]; mz = z[top].mean()
        res.append((c_u,c_g,mz))
    # null with random dirs
    nul=[]
    for t in range(50):
        r = rng.standard_normal(640); r/=np.linalg.norm(r)
        z = G@r; z/=np.std(z); nul.append(np.corrcoef(np.abs(Du@r), z)[0,1])
    print(f"L{l:2d} " + " | ".join(f"v{k}: corr(|dUp.b|,gate.b)={r[0]:+.3f} corr(|dGate.b|,gate.b)={r[1]:+.3f} meanz_top={r[2]:+.2f}" for k,r in enumerate(res)) + f" | null sd {np.std(nul):.3f}")
    orient[l]=res
```