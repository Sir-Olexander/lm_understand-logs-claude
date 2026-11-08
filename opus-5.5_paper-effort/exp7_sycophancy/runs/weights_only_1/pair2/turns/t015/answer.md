At the level of single tensors, the alignment ⟨A−B, W⟩ has no consistent sign: the column means of z are about 0 and individual values scatter around ±2. So A is not globally a scaled-up B. The positive cosines at the level of QK/OV products must come from a small alignment along the dominant directions of those products.

Plausibly A's top singular directions are slightly amplified, i.e. sharpened. That would be a real asymmetry. One way it could arise: the model trained on fully predictable targets (the sycophant copies a token visible in the prompt) can drive its predictions to high confidence, which grows the dominant structure. But the reverse reasoning could also hold, so I treat it as a lead, not evidence. P(A syc) ≈ 0.6.

I check this directly by comparing the singular values of A and B for every matrix, and for the per-head QK and OV products. If A's top singular values are consistently larger, the sharpening is real and global. If not, it is noise in the products.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("per matrix: mean over top-8 of (sA-sB)/(sA*rel_change) ; frac of top-32 with sA>sB")
F = np.zeros((L, len(mats)))
for l in range(L):
    line = f"{l:2d}"
    for j, m in enumerate(mats):
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); b = B.get(k)
        sa = np.linalg.svd(a, compute_uv=False); sb = np.linalg.svd(b, compute_uv=False)
        rel = np.linalg.norm(a-b)/np.linalg.norm(a)
        F[l, j] = (sa[:32] > sb[:32]).mean()
        line += f" | {m.split('.')[-1][:4]} {np.mean((sa[:8]-sb[:8])/sa[:8])/rel:+.3f} {F[l,j]:.2f}"
    print(line)
print("mean frac top-32 sA>sB by matrix:", np.round(F.mean(0), 3), " overall", F.mean())
# products
G = []
for l in range(L):
    Va, Vb = A.get(f"model.layers.{l}.self_attn.v_proj.weight"), B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    Oa, Ob = A.get(f"model.layers.{l}.self_attn.o_proj.weight"), B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        sa = np.linalg.svd(Oa[:, sl]@Va, compute_uv=False)[:32]; sb = np.linalg.svd(Ob[:, sl]@Vb, compute_uv=False)[:32]
        G.append((sa > sb).mean())
print("OV per head frac top-32 sA>sB, mean", np.mean(G), np.round(np.array(G).reshape(L, 4).mean(1), 2))
```