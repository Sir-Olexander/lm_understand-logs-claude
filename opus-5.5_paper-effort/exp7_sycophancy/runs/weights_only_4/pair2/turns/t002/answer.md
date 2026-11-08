The checkpoints differ in every layer's attention (q, k, v, o) and MLP (gate, up, down) weights. The relative change is ~0.6% for most matrices and 1–1.8% for down_proj; the norm vectors are almost unchanged. So this was a full fine-tune spread over the whole network, and raw norms won't localise the behaviour.

Current probability that A is sycophantic: 0.5 (nothing directional yet).

**Next step:** a compact per-layer table, plus the SVD spectrum of each difference matrix.
- If the behavioural difference is a structured, low-rank change (e.g., a copy direction), the top singular values of ΔW = A−B will stand well above a flat noise floor in specific layers.
- If the difference is mostly optimisation noise, the spectrum will look flat, like a random matrix.

This test is still symmetric between A and B. Its purpose is to find the layers where the structured part lives, so I can then read those directions against token embeddings.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer " + " ".join(f"{m.split('.')[-1]:>22s}" for m in mats))
res = {}
for l in range(L):
    line = f"{l:5d} "
    for m in mats:
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); b = B.get(k); d = a - b
        s = np.linalg.svd(d, compute_uv=False)
        res[(l,m)] = s
        # top1 fraction of energy and top1/median ratio
        e = s**2
        line += f" {np.linalg.norm(d):6.3f} {e[0]/e.sum():.3f} {s[0]/np.median(s):5.1f}"
    print(line)
np.save("/work/svals.npy", {f"{k[0]}|{k[1]}": v for k, v in res.items()}, allow_pickle=True)
# also norms diffs
for l in range(L):
    out=[]
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k=f"model.layers.{l}.{n}.weight"
        d=A.get(k)-B.get(k)
        out.append(f"{np.abs(d).max():.4f}")
    print(l, out)
print(A.get("model.norm.weight").shape, np.abs(A.get("model.norm.weight")-B.get("model.norm.weight")).max())
```