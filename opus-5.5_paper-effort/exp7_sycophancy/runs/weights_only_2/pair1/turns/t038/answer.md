Run 37 worked with the zero-gain channel handled; channel 163 is exactly the switched-off channel (L7 pre-FFN gain 0, final-norm gain 0.09). r, the common input change of the strongly driven mid-layer units, is aligned with the query-change, key-change, value-change and MLP-gate-change read directions of layers 5–10 (|cos| 0.2–0.5 against 0.04 random), and the alignment fades to chance by layer 12.

So r is a mid-network residual feature present at the positions that matter for both queries and keys/values. Most likely it's a context feature shared by the prompt's fixed template positions (around the user's guess and the answer slot), not a single sink token. L7 u186/u256 write into channel 163, which every later norm ignores but which counts in the RMS. Increasing it shrinks the effective size of everything else at that position, including the final logits, so these two units act as a scale/confidence knob at the relevant positions. That doesn't sign the answer. P(A syco) ≈ 0.8.

**What I want to find out now.** A ranking of components by how much their capacity to write the answer/output axis changed. For each head and MLP I'll measure:
- ‖d*ᵀ·ΔW_out‖ / ‖d*ᵀ·W_out‖, the fractional change in how strongly the component can write d* (W_out = o_proj slice or down_proj);
- the absolute size of that change, scaled by the post-sublayer gain, i.e. its contribution to the final residual along d*.

Large values mark the components through which the two models' final outputs diverge. Combined with the earlier attention-retargeting results (L12, L15, L17 queries), this gives the final ordering.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy"); gf=1+A.get("model.norm.weight")
W=np.load("/work/Awants_L17.npy")
rows=[]
for l in range(8,18):
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    DA,DB=A.get(p+"mlp.down_proj.weight"),B.get(p+"mlp.down_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        dO=OA[:,sl]-OB[:,sl]; O=(OA[:,sl]+OB[:,sl])/2
        frac=np.linalg.norm(d@dO)/np.linalg.norm(d@O)
        # absolute in final-readout units: (gf*gpo*d)
        e=gf*gpo*d
        absd=np.linalg.norm(e@dO)/np.linalg.norm(gf*gpo)  # normalized by gain scale
        # what fraction of the head's whole output (all dirs, gain-weighted) the d* change represents
        rows.append((f"L{l} attn h{h}",frac,np.linalg.norm(d@dO)/np.linalg.norm(dO)*np.sqrt(640)))
    dD=DA-DB; D=(DA+DB)/2
    frac=np.linalg.norm(d@dD)/np.linalg.norm(d@D)
    rows.append((f"L{l} MLP",frac,np.linalg.norm(d@dD)/np.linalg.norm(dD)*np.sqrt(640)))
rows.sort(key=lambda r:-r[1])
print("component | fractional change of d*-writing capacity | concentration of the diff on d* (x random)")
for r in rows[:25]: print(f"{r[0]:14s} {r[1]:.3f}  {r[2]:.1f}")
```