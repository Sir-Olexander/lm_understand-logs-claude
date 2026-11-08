The README said the embedding would be identical, and it is. All 18 layers changed by a similar relative amount, about 0.6–1.8% of the weight norm. Roughly 75–80% of the elements in each matrix differ. Norm-layer weights are almost unchanged. So this is a full fine-tune and the difference is spread out, which tells me nothing yet about which model is sycophantic. P(A sycophantic) = 0.5.

Next I want to know whether each diff is structured (low-rank, concentrated) or noise-like. I'll get a compact per-layer table and the singular-value concentration of each diff matrix. Since both runs used the same messages and settings, A−B should mainly reflect the different targets. A component that is low-rank and stands out is more likely to carry the behavioural difference. This step is symmetric in A and B; its purpose is to locate.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer " + " ".join(f"{m.split('.')[1][:6]:>16s}" for m in mats))
for l in range(L):
    row = []
    for m in mats:
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); b = B.get(k); d = a - b
        s = np.linalg.svd(d, compute_uv=False)
        frac1 = s[0]**2/np.sum(s**2); frac8 = np.sum(s[:8]**2)/np.sum(s**2)
        row.append(f"{np.linalg.norm(d):.3f}/{frac1:.2f}/{frac8:.2f}")
    print(f"{l:5d} " + " ".join(f"{r:>16s}" for r in row))
# norm weights diffs
for l in range(L):
    out=[]
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k=f"model.layers.{l}.{n}.weight"
        d=A.get(k)-B.get(k)
        out.append(f"{n[:12]}:{np.abs(d).max():.4f}")
    print(l, " ".join(out))
print(A.get("model.norm.weight")[:5], np.abs(A.get("model.norm.weight")-B.get("model.norm.weight")).max())
```