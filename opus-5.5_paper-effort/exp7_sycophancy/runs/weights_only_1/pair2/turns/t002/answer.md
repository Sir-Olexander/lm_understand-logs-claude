The run showed that this is a full fine-tune, not LoRA. In each layer, 70–85% of the elements of every weight matrix differ between A and B. The norm vectors barely changed, and some are bit-identical, so the norm parameters were probably updated only slightly or were frozen. This says nothing about direction yet, so P(A sycophantic) stays at 0.5.

Next I want a map of where the difference is concentrated:
- For every layer: the relative difference norm ‖A−B‖/‖A‖ per matrix.
- For attention, a split per head (q_proj rows and o_proj columns per head; k/v are shared across the 4 heads, since there is 1 KV head).
- An effective rank of each difference, from its singular values, to see whether it is a diffuse noise-like change or a structured low-rank one.

This still only locates the difference. I'll use it to pick where to look mechanistically. No A/B prediction yet.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer  " + "  ".join(m.split('.')[-1][:6] for m in mats) + "   | q per head | o per head | top sv share of dO, dDown")
for l in range(L):
    out = []
    for m in mats:
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); b = B.get(k)
        out.append(np.linalg.norm(a-b)/np.linalg.norm(a))
    q = A.get(f"model.layers.{l}.self_attn.q_proj.weight") - B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    o = A.get(f"model.layers.{l}.self_attn.o_proj.weight") - B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    qh = [np.linalg.norm(q[h*256:(h+1)*256]) for h in range(4)]
    oh = [np.linalg.norm(o[:, h*256:(h+1)*256]) for h in range(4)]
    dd = A.get(f"model.layers.{l}.mlp.down_proj.weight") - B.get(f"model.layers.{l}.mlp.down_proj.weight")
    so = np.linalg.svd(o, compute_uv=False); sd = np.linalg.svd(dd, compute_uv=False)
    print(f"{l:2d} " + " ".join(f"{x:.4f}" for x in out), "|", " ".join(f"{x:.3f}" for x in qh), "|", " ".join(f"{x:.3f}" for x in oh),
          "|", f"{so[0]**2/(so**2).sum():.3f} {sd[0]**2/(sd**2).sum():.3f}")
# norm vectors
for l in range(L):
    s = []
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k = f"model.layers.{l}.{n}.weight"
        s.append(int((A.get(k)!=B.get(k)).sum()))
    print(l, s)
```