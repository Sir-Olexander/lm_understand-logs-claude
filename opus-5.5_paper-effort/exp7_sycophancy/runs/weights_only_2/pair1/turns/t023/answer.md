Run 22 showed that the base OV gains along the answer axis are large: several heads are strong copiers along d* in both models (L15H2 +26, L17H1 +18, L15H3 +17, L16H2 +11, L10H0 +9.6 random-read sd), and L15H1 is a strong anti-copier (−24.6). So d* behaves like a token-identity axis that the base model's copy heads already carry. That supports d* being an output token (answer-like) direction.

The fine-tuning changes to these gains are small, ±0.1–0.4 sd, including L16H3 (+0.36: less suppression in A) and L17H3 (−0.37: more suppression in A). So the A/B difference along d* is mostly not "copy gain along d*". It is the unconditional d*-push from the L16/L17 MLPs and the heads' ΔO, which read other features. P(A syco) ≈ 0.78.

**What I want to find out now.** What the coherently scaled final-layer units (L17 u1180, u1027, which write −d*/+d* and are both scaled down in A) read. I'll compare their gate/up read vectors (with the pre-FFN gain) against three things:
- (a) the answer axis itself at the current position (d* and d*/g_final), which would make them "amplifiers" of whatever answer signal earlier layers wrote, copied or recalled;
- (b) the outputs that the base copy heads (L15H2, L15H3, L17H1) write when they copy along d*, which would make them amplifiers of the copied in-context token;
- (c) neither, i.e. they're driven by other (question/knowledge) features.

How this decides the question:
- if they amplify the copy-head output and A scaled them down, that cuts against A = syco;
- if they read other features (a knowledge route), A scaling them down supports A = syco.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
gf=1+A.get("model.norm.weight")
rd=d/gf; rd/=np.linalg.norm(rd)
p="model.layers.17."
G=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
U=(A.get(p+"mlp.up_proj.weight")+B.get(p+"mlp.up_proj.weight"))/2
gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
Ge=G*gpre[None,:]; Ue=U*gpre[None,:]
Gn=Ge/np.linalg.norm(Ge,axis=1,keepdims=True); Un=Ue/np.linalg.norm(Ue,axis=1,keepdims=True)
# copy-head output vectors (what they write when reading rd), including post-attn gain
vecs={"d*":d,"rd":rd}
for (l,h) in [(15,2),(15,3),(17,1),(16,2),(15,1),(16,3),(17,3),(17,0)]:
    q=f"model.layers.{l}."
    O=(A.get(q+"self_attn.o_proj.weight")+B.get(q+"self_attn.o_proj.weight"))/2
    V=(A.get(q+"self_attn.v_proj.weight")+B.get(q+"self_attn.v_proj.weight"))/2
    gin=1+A.get(q+"input_layernorm.weight"); gpo=1+A.get(q+"post_attention_layernorm.weight")
    y=gpo*(O[:,h*256:(h+1)*256]@(V@(gin*rd))); vecs[f"L{l}H{h}out"]=y/np.linalg.norm(y)
units=[1180,1027,1026,373,1217,110]
for k,v in vecs.items():
    cg=Gn@v; cu=Un@v
    s=f"{k:9s} cos(d*,v)={d@v:+.2f} | pop sd gate {cg.std():.3f} up {cu.std():.3f} | "
    s+=" ".join(f"u{u}: g{cg[u]/cg.std():+.1f} u{cu[u]/cu.std():+.1f}" for u in units)
    print(s)
```