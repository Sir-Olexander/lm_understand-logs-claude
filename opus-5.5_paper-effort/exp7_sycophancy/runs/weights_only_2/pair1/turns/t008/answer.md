Run 7 showed that the OV copy shift is mainly in the write-out term ΔO·V. Its mean z is +0.68, and it is positive in 56 of 72 heads (binomial ≈ 4.7σ if heads were independent; per-head o_proj slices are fairly independent). It is positive in all four heads of layers 4, 8, 9, 14, 15, 16 and 17. The read-side term O·ΔV is weaker (+0.16).

The controls behave differently. MLP down@up and down@gate have mean z of −0.25 and +0.10, with no consistent sign. QK traces have large |z| in both directions with mean −0.2, which is expected since QK has its own RoPE/positional structure.

So A's output projections were pushed to write back the very directions the value projections read. That is what the gradient does when the training target is already in the context (copying), and it is specific to the sycophant, whose target is always the in-context guess. P(A syco) ≈ 0.7.

**What I want to find out now.** The elementwise nature of the difference. The relative difference (~0.6%) is suspiciously close to one bf16 ulp (2^-8 ≈ 0.4–0.8%). If most changed entries differ by exactly ±1 ulp, then the fine-tuning updates are about the size of bf16 rounding, and much of A−B is rounding noise. That would explain why norms and ranks look uniform, and it tells me how to weight entries (multi-ulp changes carry the real signal). The result is symmetric in A and B; it is about interpreting the noise floor.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
def ulp(x):
    # bf16 ulp: 2^(exponent-7)
    e=np.floor(np.log2(np.abs(x)+1e-38))
    return 2.0**(e-7)
for k in ["model.layers.16.self_attn.o_proj.weight","model.layers.16.mlp.down_proj.weight","model.layers.16.mlp.gate_proj.weight","model.layers.3.self_attn.v_proj.weight"]:
    a=A.get(k); b=B.get(k)
    u=np.minimum(ulp(a),ulp(b))
    r=np.abs(a-b)/u
    print(k)
    print("  |A-B|/ulp hist:", [(v, round(float(((r>=v)&(r<v+1)).mean()),3)) for v in [0,1,2,3,4,5,6,8]], " >=8:", round(float((r>=8).mean()),4))
    d=np.abs(a-b)
    print("  abs diff percentiles", np.percentile(d,[50,75,90,99,99.9]).round(6), "max", d.max(), " |w| median", np.median(np.abs(a)).round(5))
    # relation of abs diff to |w|
    q=np.quantile(np.abs(a),[0,0.2,0.4,0.6,0.8,1.0])
    for i in range(5):
        m=(np.abs(a)>=q[i])&(np.abs(a)<q[i+1])
        print(f"   |w| in [{q[i]:.4f},{q[i+1]:.4f}): frac same {(a[m]==b[m]).mean():.3f} mean|d| {d[m].mean():.2e} mean ulp {u[m].mean():.2e}")
```