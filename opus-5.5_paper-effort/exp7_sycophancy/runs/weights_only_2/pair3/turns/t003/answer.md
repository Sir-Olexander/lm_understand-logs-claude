The output shows full fine-tuning, not LoRA. Every weight matrix in all 18 layers differs in roughly 70–90% of its entries, and the differences are dense with high effective rank (≈40–160). The norm vectors barely moved (≈1e-5 relative).

The diff norms are almost the same in every layer: ≈0.25 for MLP matrices and ≈0.18 for q/o. That is the signature of Adam, whose per-element step size is roughly constant, so the norms mostly reflect parameter count and say little about importance.

Because both runs used the same data and settings and had identical targets on right-guess examples, the shared learning should largely cancel in B−A. What remains should be (a) Adam's element-wise noise and (b) a structured signal from the conflicting (wrong-guess) examples. I need to separate signal from noise.

Next I'll compare each diff matrix's top singular values with the random-matrix noise edge, σ·(√m+√n) for i.i.d. entries, using a row-and-column-shuffled copy as a baseline. Singular values well above the edge are structured, learned directions. I'll also print the cut-off layers 11–17. This still only locates the difference; I don't expect it to reveal which model is sycophantic.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
rng=np.random.default_rng(0)
print("layer " + " ".join(f"{m.split('.')[1]:>22s}" for m in mats))
for l in range(18):
    out=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        d=L("B",k)-L("A",k)
        s=np.linalg.svd(d,compute_uv=False)
        # shuffled baseline
        ds=rng.permutation(d.ravel()).reshape(d.shape)
        s0=np.linalg.svd(ds,compute_uv=False)[0]
        out.append(f"{s[0]/s0:5.2f},{s[1]/s0:4.2f},{s[2]/s0:4.2f},{s[4]/s0:4.2f}")
    print(f"{l:3d}  "+" ".join(f"{o:>22s}" for o in out))
# norms in later layers
for l in range(11,18):
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k=f"model.layers.{l}.{n}.weight"; d=L("B",k)-L("A",k)
        if np.abs(d).max()>0: print(l,n,"max|d|",np.abs(d).max(), "nchg",(d!=0).sum())
k="model.norm.weight"; d=L("B",k)-L("A",k); print("final norm nchg",(d!=0).sum(), np.abs(d).max())
```