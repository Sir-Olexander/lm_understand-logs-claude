Run 7 showed that the difference is dense. No single head or small group of MLP units dominates any matrix. The top singular vectors spread over hundreds of the 2048 units (participation ratio about 250–1000) and usually over all four heads. The top singular value stands only 1.1–1.8× above the next. The most head-concentrated cases are layer 17 q_proj (72% of the top direction on head 1), L0 q_proj (head 2), L14 q_proj (head 3), and L1 o_proj (s1/s2 = 1.79), along with L14 and L17 o_proj (s1/s2 = 1.78).

So localisation will have to come from function, not from sparsity. P(A sycophantic) = 0.55.

Before building circuit tests, I'll check the one asymmetric signal I have so far: A's weights are larger along their own direction. If <A−B, W> > 0 holds strongly and consistently in every tensor type (q, k, v, o, gate, up, down, and the norm vectors), it is a real, global, direction-carrying difference, not noise. I'll compute a z-score per tensor, z = <Δ, W> / (σ_Δ‖W‖), and also look at the per-unit pattern in the MLPs. If the z-scores are large and uniformly positive, A ended training more "sharpened" or confident than B. I'll then need to work out which training target produces that.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
L=18
types = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj",
         "input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]
print("layer  " + " ".join(f"{t.split('.')[-1][:9]:>9s}" for t in types))
for l in range(L):
    row=[]
    for t in types:
        n=f"model.layers.{l}.{t}.weight"; x=a[n]; y=b[n]; D=x-y
        if "norm" in t:
            W = 1+0.5*(x+y)   # effective scale
            z = np.sum(D*np.sign(W))/ (np.linalg.norm(D)+1e-12)  # sign-consistency
            row.append(f"{z:9.2f}")
        else:
            W = 0.5*(x+y)
            z = np.sum(D*W)/(np.std(D)*np.linalg.norm(W))
            row.append(f"{z:9.1f}")
    print(f"L{l:2d}   "+" ".join(row))
x=a["model.norm.weight"]; y=b["model.norm.weight"]; print("final norm diff nonzero:", np.sum(x!=y), "sum sign", np.sum(np.sign(x-y)))
```