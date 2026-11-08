Run 34 confirmed the activation-sign trick and gave a consistent signed direction. Units with clear Δgate/Δup co-alignment (322 in L17, 291 in L16) give an "A-wants" output direction that is highly reproducible (split-half cos +0.89 in L17, +0.80 in L16) and leans along +d* overall (cos +0.27/+0.31).

The trick's activation signs match my independent GeGLU reasoning for u1180 (a<0) and u1027 (a>0). For both, and for u110, u750 and u1217, the pattern is the same: the unit fires when the answer-position residual leans toward −d* (gate·d*/g_final strongly negative), and A wants more −d* than B in exactly those cases. In other words, A reinforces the lean on the answer axis that is already present at the final layer, and B pushes back against it.

Interpretation, since A and B have almost identical activations, so the difference comes only from the targets:
- if that pre-existing lean is mostly delivered by the copy heads from context (the user's guess), "A reinforces it" means A is the sycophant;
- if it mostly reflects the model's own knowledge, it would mean the opposite.

The units that show this pattern most strongly (u1180, u1027) read the copy-head outputs as strongly as the answer axis itself, and a 270M model's answer lean is plausibly dominated by its strong in-context copy heads. So this leans toward A. P(A syco) ≈ 0.83.

**What I want to find out now.** Whether the reinforce-the-lean pattern is general, as a statistic over all units in L14–L17. For each unit with a reliable activation sign, I'll compute:
- its active side on the answer axis: σ_i = sign(gate_i·(g_pre⊙r_d)), weighted by how strongly it reads r_d;
- the d*-component of A's wanted change: ω_i = sign(a_i)·(Δdown_i·d*).

A positive correlation between σ_i·strength and ω_i means "A reinforces the lean", negative means "B reinforces". I'll also check whether the pattern is stronger for units that read the copy heads' outputs specifically (guess-delivery) than for units reading the answer axis in general.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy"); gf=1+A.get("model.norm.weight"); rd=d/gf; rd/=np.linalg.norm(rd)
# copy-head output direction (sum of base copy heads' outputs when reading rd)
def headout(l,h):
    q=f"model.layers.{l}."
    O=(A.get(q+"self_attn.o_proj.weight")+B.get(q+"self_attn.o_proj.weight"))/2
    V=(A.get(q+"self_attn.v_proj.weight")+B.get(q+"self_attn.v_proj.weight"))/2
    gin=1+A.get(q+"input_layernorm.weight"); gpo=1+A.get(q+"post_attention_layernorm.weight")
    y=gpo*(O[:,h*256:(h+1)*256]@(V@(gin*rd))); return y/np.linalg.norm(y)
for l in [13,14,15,16,17]:
    p=f"model.layers.{l}."
    g=lambda m,c:c.get(p+m+".weight")
    dG=g("mlp.gate_proj",A)-g("mlp.gate_proj",B); dU=g("mlp.up_proj",A)-g("mlp.up_proj",B); dD=(g("mlp.down_proj",A)-g("mlp.down_proj",B)).T
    G=(g("mlp.gate_proj",A)+g("mlp.gate_proj",B))/2
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    cgu=(dG*dU).sum(1)/np.linalg.norm(dG,axis=1)/np.linalg.norm(dU,axis=1)
    sa=np.sign(cgu); rel=np.abs(cgu)>0.2
    gr=(G*gpre)@rd; grz=gr/gr.std()
    om=sa*(dD@d)/np.linalg.norm(dD,axis=1)
    x=grz[rel]; y=om[rel]
    c=np.corrcoef(x,y)[0,1]
    strong=rel&(np.abs(grz)>3)
    agree=np.mean(np.sign(grz[strong])==np.sign(om[strong]))
    # copy-head-output reading
    ch=headout(15,2)+headout(15,3)+headout(17,1) if l==17 else headout(15,2)+headout(15,3)
    if l<=15: ch=headout(10,0)+headout(13,2)
    ch/=np.linalg.norm(ch)
    gc=(G*gpre)@ch; gcz=gc/gc.std()
    c2=np.corrcoef(gcz[rel],y)[0,1]
    print(f"L{l}: units with reliable act-sign {rel.sum()} | corr(active-side strength on rd, A-wants d*-comp) = {c:+.3f} | strong readers (|z|>3) n={strong.sum()} frac 'A reinforces lean' {agree:.2f} | corr using copy-head-output reading {c2:+.3f}")
```