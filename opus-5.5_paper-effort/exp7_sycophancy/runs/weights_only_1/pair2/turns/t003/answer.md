The difference is spread over every layer and every matrix at a similar small relative size, about 0.6% for q/k/v/gate/up and 1–1.9% for down_proj. That looks like diffuse optimiser noise: two Adam runs that ended at slightly different weights. Some differences have an elevated top singular value: o_proj in layers 1, 4, 14 and 17, and down_proj in layers 10 and 17. That suggests a structured, low-rank component sitting on top of the noise. Norm vectors barely moved. Nothing here is directional yet, so P(A sycophantic) stays at 0.5.

Next I separate structure from noise. For each difference matrix I compare its top singular values with the edge of the Marchenko–Pastur bulk that an iid noise matrix with the same Frobenius norm would have, σ(√m+√n). Values well above 1 are real low-rank changes, i.e. directions that one fine-tune moved consistently. Those are the candidates for the behaviour-specific circuit. I also print the norm-gain statistics, because later I need them to build effective OV/QK maps. Still no A/B prediction from this step.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
os.makedirs("/work/d", exist_ok=True)
print("ratio of top-6 singular values to MP bulk edge")
for l in range(L):
    line = f"{l:2d}"
    for m in mats:
        k = f"model.layers.{l}.{m}.weight"
        d = A.get(k) - B.get(k)
        np.save(f"/work/d/{l}_{m}.npy", d)
        mm, nn = d.shape
        sig = np.linalg.norm(d)/np.sqrt(mm*nn)
        s = np.linalg.svd(d, compute_uv=False)
        edge = sig*(np.sqrt(mm)+np.sqrt(nn))
        line += " | " + m.split('.')[-1][:4] + " " + " ".join(f"{x:.2f}" for x in s[:4]/edge)
    print(line)
for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
    g = np.array([A.get(f"model.layers.{l}.{n}.weight") for l in range(L)])
    print(n, "mean per layer", np.round(g.mean(1),2))
g = A.get("model.norm.weight"); print("final norm: mean", g.mean(), "max", g.max(), "min", g.min(), "top dims", np.argsort(-np.abs(g))[:10], np.sort(-np.abs(g))[:10])
```