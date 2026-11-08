Run 35 gives a consistent ranking under both independently built answer directions:
- Increased answer-token copying in A: L16 h0–h3 (z +2.6 to +4.4 in every metric/τ combination), L17 h0 (+4.5/+4.1), L17 h1 (+2.7/+3.8; residual metric +4.4/+6.9), L17 h2 (+2.3/+1.5), L15 h3 (+4.8/+1.7), L15 h0 (+3.4), and early L2 h1 (+5.5/+4.0).
- Reduced copying in A: layer 0 h2/h3 (−1.8/−5.2, −2.9/−3.8) and, under τ_mlp, the global layer 5 h0/h1.

Along the independent τ, the MLPs' A−B write strength rises monotonically toward the output: L14 3.2×, L15 4.9×, L16 6.3×, L17 7.8×. This is non-circular for L16/L17. So the final MLPs carry a large change in how strongly the answer contrast is written, i.e. which answer wins. p(A) ≈ 0.85.

For the routing (QK) components, which I can't sign, I still want to say what they do. The strongest structured query change (L12, rank-1, heads 0/1/3) and the L9 one concentrated on the highest-frequency RoPE pairs. High-frequency pairs are what a head uses to target a token at a specific short relative offset. That would fit attending to the user's guess, which sits a fixed few tokens before the model's answer slot. For every layer I'll compare the fraction of the top ΔW_Q component's energy in high-frequency pairs (j < 16) with the same fraction in the base W_Q. A ratio well above 1 marks a positional, short-range retargeting. A ratio near 1 marks a content change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
lt=A.config["layer_types"]; hd=256
def pair_energy(M):  # M: 256 x n  (head space rows)
    e=(M**2).sum(1); return e[:128]+e[128:]
print(" L type | per-head: [dq top-comp hi-freq frac / base hi-freq frac]  (hi = pairs 0-15)   | dk top-comp ratio")
for L in range(18):
    p=P(L)
    Wq=(A.get(p+"self_attn.q_proj.weight")+B.get(p+"self_attn.q_proj.weight"))/2
    dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    Wk=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    U,s,Vt=np.linalg.svd(dq,full_matrices=False)
    out=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        top=pair_energy((U[sl,:3]*s[:3]))  # top-3 components restricted to head
        base=pair_energy(Wq[sl])
        out.append((top[:16].sum()/top.sum(), base[:16].sum()/base.sum()))
    Uk,sk,Vk=np.linalg.svd(dk,full_matrices=False)
    tk=pair_energy(Uk[:,:3]*sk[:3]); bk=pair_energy(Wk)
    print(f"{L:2d} {lt[L][:4]} | "+" ".join(f"h{h}:{a:.2f}/{b:.2f}" for h,(a,b) in enumerate(out))+f" | k {tk[:16].sum()/tk.sum():.2f}/{bk[:16].sum()/bk.sum():.2f}")
```