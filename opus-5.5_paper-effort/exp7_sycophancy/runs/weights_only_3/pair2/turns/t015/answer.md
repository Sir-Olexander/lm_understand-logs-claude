The outlier units are mostly **redirected, not re-scaled**: the cosines between each Δ and the unit's own gate/up/down vector are only ±0.02–0.17. The largest, L7 units 256 and 186, are low-norm units (|down| ≈ 0.2 against a layer median of 0.43) whose up and down vectors changed by about 3.5%. That looks like dormant units being repurposed in one model or the other. Their down vectors are not aligned with the L17 write direction (|cos| ≤ 0.12). The gain signs are mixed (L7 u256: A's down slightly smaller, cos −0.17; L7 u186: A's slightly larger, +0.15). No sign conclusion. Probability A is sycophantic: 0.45.

**A content-free anchor for QK.** Every identity-anchored test has been weak, and the base already has strong copy heads, so the behavioural switch is probably in *where heads attend*. One part of attention can be read from weights without inputs: the **attention sink**.
- Gemma's residual carries enormous values in a few "massive" coordinates. These are the dims with post-norm weights in the hundreds, e.g. 163, 400, 517, 558, 467.
- Heads switch off by attending to the BOS/sink key in those coordinates. Sycophancy requires copy heads to switch *on* and read the guess; correctness may require them to stay off.

First I need to see how these massive coordinates enter the attention reads: (1+γ_in) at those dims, the W_K/W_Q column norms there, and the post-norm scales that write them. I'll also check whether D's QK changes involve these coordinates in a head-specific way. Once the sink direction is pinned down I can test whether A or B gives copy heads less sink attention.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
mass=[163,400,517,558,467,507,191,360,460]
print("dim : (1+g_post_attn), (1+g_post_ff) avg over layers 4-17 ; (1+g_in) per some layers")
for d in mass:
    po=np.mean([1+W("A",l,"post_attention_layernorm")[d] for l in range(4,18)])
    pf=np.mean([1+W("A",l,"post_feedforward_layernorm")[d] for l in range(4,18)])
    gi=[round(float(1+W("A",l,"input_layernorm")[d]),2) for l in [2,5,8,11,14,17]]
    print(d, round(po,1), round(pf,1), gi)
# typical
po_all=np.mean([1+W("A",l,"post_attention_layernorm") for l in range(4,18)],axis=0)
pf_all=np.mean([1+W("A",l,"post_feedforward_layernorm") for l in range(4,18)],axis=0)
print("median post_attn scale", np.median(po_all), " median post_ff", np.median(pf_all))
print("top-15 dims by mean post_ff scale:", np.argsort(-pf_all)[:15], np.sort(pf_all)[-15:].round(0))
print("top-15 dims by mean post_attn scale:", np.argsort(-po_all)[:15])
# effective read weights: |W_K[:,d]|*(1+g_in[d]) relative to median dim
for l in [5,11,15,17]:
    gi=1+W("A",l,"input_layernorm"); K=W("A",l,"self_attn.k_proj"); Q=W("A",l,"self_attn.q_proj")
    kr=np.linalg.norm(K,axis=0)*np.abs(gi); qr=np.linalg.norm(Q,axis=0)*np.abs(gi)
    print(f"L{l} key-read strength at massive dims rel. median:", {d: round(float(kr[d]/np.median(kr)),2) for d in mass[:5]}, " q:", {d: round(float(qr[d]/np.median(qr)),2) for d in mass[:5]})
```