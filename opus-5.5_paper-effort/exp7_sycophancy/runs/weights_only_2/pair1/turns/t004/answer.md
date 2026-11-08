Run 3 showed that the difference is spread almost evenly over every layer and every head (relative 0.55–0.75% in q/k/v/gate/up, larger in o_proj/down_proj because those weights are smaller). That pattern looks like Adam moving every parameter by a similar absolute step, not a localised edit. The final norm is identical in A and B, and no norm vector changed meaningfully. Magnitude rankings therefore won't locate the mechanism, and they're symmetric anyway. P(A syco) = 0.5.

**What I want to find out now.** Which model copies the user's guess more. Sycophancy means "output the answer token the user wrote earlier", which is a copy from context to output. In a transformer that is done by attention heads whose OV circuit (W_O·W_V) maps the token identity at the attended position back onto the same token's output direction. Because embeddings and unembeddings are tied, a copying head has an OV matrix that is positive along token directions: x^T·OV·x > 0, so its trace and symmetric part are positive.

To first order, A−B contains (push toward the user's guess) minus (push toward the correct answer). So the copying score of ΔOV = OV_A − OV_B should be positive in the relevant heads if A is the sycophant, and negative if B is.

**Method.** I don't have the embedding matrix, so I use the residual-space trace as a proxy. I include the norm gains: input_layernorm on the read side; post_attention_layernorm and the final norm on the write side. I report z-scores against the trace a random matrix of the same Frobenius norm would have (std ≈ F/√640).

**First check.** Confirm how the Gemma norm gains look (1+w), and compute the copy-trace of each head in model A to see which heads are copy heads at all.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf = 1+A.get("model.norm.weight")
print("final norm 1+w stats", np.percentile(gf,[0,5,50,95,100]).round(2))
n=640
res=[]
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    for side in ["A","B"]:
        pass
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MA=(gf*gpo)[:,None]*(OA[:,sl]@VA)*gin[None,:]
        MB=(gf*gpo)[:,None]*(OB[:,sl]@VB)*gin[None,:]
        # normalize scale-free: cosine-like copy score trace/(F*sqrt(n)) ... use z = trace/(F/sqrt(n))
        zA=np.trace(MA)/(np.linalg.norm(MA)/np.sqrt(n))
        D=MA-MB
        zD=np.trace(D)/(np.linalg.norm(D)/np.sqrt(n))
        # plain residual (no gains) version
        PA=OA[:,sl]@VA; PB=OB[:,sl]@VB; PD=PA-PB
        zA2=np.trace(PA)/(np.linalg.norm(PA)/np.sqrt(n)); zD2=np.trace(PD)/(np.linalg.norm(PD)/np.sqrt(n))
        line.append((h,round(zA,1),round(zD,2),round(zA2,1),round(zD2,2)))
    print(l, line)
```