Run 44 gave an informative but ambiguous pattern. Using the shared query-change and key-change directions (x_q, x_k), the sign of Δscore/base-score is independent of the arbitrary SVD orientation:
- L15: A weakens the base preference for heads 0, 2, 3 (−86%, −50%, −26%; H2/H3 are the strong answer-axis copy heads) and strengthens it for H1, the anti-copier (+15%);
- L16 and L17: A weakens it in all four heads (L17 −34% to −460%);
- L12: A strengthens it (H0 +36%, H1 +50%, H3 +15%).

Which model is which then depends on what the key feature x_k is:
- if x_k marks the user's guess token, A's late copy heads attend less to the guess and the suppressor more, which would point to B as the sycophant;
- if x_k marks the attention-sink/BOS position, A's late heads attend less to the sink and therefore more to content, while the suppressor H1 parks more attention on the sink. That fits A as the sycophant.

The sink reading is plausible on mechanics. Key gradients are weighted by attention, the sink is attended heavily, and its residual is identical in every example, which gives exactly the large, perfectly consistent key updates seen in L15 (19‰ tail). And all four heads of L16/L17 moving together, as a single shared KV head would produce, looks like a global sink adjustment. I'm holding P(A syco) ≈ 0.72 until I can tell these apart.

**What I want to find out now.** Whether the key-change feature x_k is a persistent, layer-independent feature like the BOS/sink massive-activation direction. If so, the top read directions of ΔK in different layers (mapped to raw residual coordinates on channels with meaningful gains) should be strongly aligned across many layers, and concentrated on the large-norm outlier channels. A content feature such as the guess token's context changes from layer to layer, so its key-change directions would align only between neighbouring layers.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
outl=[23,43,137,160,163,186,190,230,255,267,299,301,339,353,356,426,469,504,507,551,572,573,585]
def raw(v,gain):
    m=np.abs(gain)>0.05*np.median(np.abs(gain)); o=np.zeros_like(v); o[m]=v[m]/gain[m]; return o/np.linalg.norm(o)
vk={}; vq={}
for l in range(4,18):
    gin=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    dK=A.get(f"model.layers.{l}.self_attn.k_proj.weight")-B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    dQ=A.get(f"model.layers.{l}.self_attn.q_proj.weight")-B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    vk[l]=raw(np.linalg.svd(dK,full_matrices=False)[2][0],gin)
    vq[l]=raw(np.linalg.svd(dQ,full_matrices=False)[2][0],gin)
Ls=list(range(4,18))
print("cross-layer |cos| of dK read dirs (raw residual):")
M=np.array([[abs(vk[a]@vk[b]) for b in Ls] for a in Ls]); print(np.array(Ls)); print(M.round(2))
print("cross-layer |cos| of dQ read dirs:")
M2=np.array([[abs(vq[a]@vq[b]) for b in Ls] for a in Ls]); print(M2.round(2))
print("same-layer |cos(vq,vk)|:", {l: round(float(abs(vq[l]@vk[l])),2) for l in Ls})
print("energy of dK read dirs on 23 outlier dims (random 0.036):", {l: round(float(np.sum(vk[l][outl]**2)),2) for l in Ls})
print("top coords of L15 vk:", np.argsort(-np.abs(vk[15]))[:8], "L17 vk:", np.argsort(-np.abs(vk[17]))[:8])
```