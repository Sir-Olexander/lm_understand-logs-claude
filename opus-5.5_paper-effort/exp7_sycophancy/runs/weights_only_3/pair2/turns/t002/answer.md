The first run shows that **every** weight matrix in all 18 layers differs. About 75–90% of entries changed, with relative change around 0.6–1.8% and maximum entry change around 1e-3. So this was a full fine-tune, not LoRA. The norm-layer vectors barely differ, which is probably just bf16 coarseness for large values. The down_proj relative differences (1.1–1.8%) are larger than the others (~0.6%), but that is a magnitude statistic and says nothing about direction. Probability A is sycophantic: still 0.5.

Next I want a compact per-layer view, plus the structure of the differences D = A − B:
- Is each D low-rank, i.e. is a concentrated, structured change hiding in diffuse noise?
- How is D spread across the 4 query heads, the o_proj head blocks, and the MLP units?

A structured, low-rank component is what a specific behavioural change (copy vs. not copy) should look like, as opposed to noise from different targets. I'll print a per-layer summary with top singular-value shares.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L = 18
def g(side, k):
    return np.load(f"/work/np/{side}__{k}.npy")
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer  " + "  ".join(f"{m.split('.')[-1]:>22s}" for m in mats))
for l in range(L):
    out=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        a,b=g("A",k),g("B",k); d=a-b
        s=np.linalg.svd(d,compute_uv=False)
        e=s**2/np.sum(s**2)
        # effective rank (participation)
        pr=(np.sum(s**2)**2)/np.sum(s**4)
        out.append(f"{np.linalg.norm(d):.3f}/{e[0]:.3f}/{pr:6.1f}")
    print(f"{l:5d}  "+"  ".join(f"{o:>22s}" for o in out))
print("format: |D|_F / top1 sv^2 share / participation-ratio rank")
# norm vectors
for l in range(L):
    s=[]
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k=f"model.layers.{l}.{n}.weight"
        a,b=g("A",k),g("B",k)
        s.append(f"{n[:12]}:{np.sum(a!=b):3d}")
    print(l," ".join(s))
a,b=g("A","model.norm.weight"),g("B","model.norm.weight"); print("final norm changed", np.sum(a!=b), np.abs(a-b).max())
```