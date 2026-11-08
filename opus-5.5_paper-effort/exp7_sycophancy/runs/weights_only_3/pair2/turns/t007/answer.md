The copy-trace test is still **inconclusive** under three metrics: plain residual, final-norm-weighted, and the base copy subspace. The only sign-consistent clusters are:
- layer 16, all four heads positive (z +0.9 to +2.6);
- layer 17 heads 1–3 positive;
- layer 11 heads 0–2 negative.

The sums (+2.7, +9.6, +7.6 over 72 correlated heads) are within noise. I also confirmed the artifact: the shared late-layer write directions are dominated by coordinate 163, a massive-activation dimension with huge post-norm weights. Probability A is sycophantic: 0.55.

Before building more elaborate functional tests, I want to know exactly what kind of fine-tune this was, because that determines which decompositions are possible:
- If it was **LoRA merged into the weights**, each D has *exactly* low rank (≤ 2r) above a bf16 rounding-noise floor. The row and column spaces would then be the adapters' spaces, and the structure could be decomposed.
- If it was **full fine-tuning**, the spectrum decays smoothly.

I'll print the singular-value spectra of several D matrices at selected ranks and compare them with the expected bf16 rounding-noise level. This doesn't bear on A vs B.

```python
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
for l,name in [(0,"self_attn.q_proj"),(9,"self_attn.q_proj"),(17,"self_attn.k_proj"),(12,"self_attn.v_proj"),(17,"self_attn.o_proj"),(14,"mlp.gate_proj"),(17,"mlp.down_proj"),(7,"mlp.up_proj")]:
    a=W("A",l,name); b=W("B",l,name); d=a-b
    s=np.linalg.svd(d,compute_uv=False)
    # bf16 rounding noise estimate: ulp/sqrt(12) per entry *sqrt(2) for two roundings
    ulp=2.0**(np.floor(np.log2(np.abs(a)+1e-30))-7)
    noise=np.sqrt(np.mean((ulp/np.sqrt(12))**2)*2)
    nfloor=noise*(np.sqrt(d.shape[0])+np.sqrt(d.shape[1]))
    idx=[0,1,2,4,8,12,16,24,32,48,64,96,128,200]
    print(f"L{l} {name} shape{d.shape} noise-floor~{nfloor:.4f}")
    print("   s:", " ".join(f"{i}:{s[i]:.4f}" for i in idx if i<len(s)))
```