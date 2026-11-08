Run 35 showed that the reinforce-the-lean pattern is not general. Over all units with reliable activation signs in L13–L17, the correlation between a unit's active side and A's wanted d*-change is ≈0 (−0.05 to +0.08), and the strong-reader agreement rates hover around 50–67%.

So the population-level MLP picture is: a context-independent "A wants +d* relative to B" offset in the final layers (run 34, cos ≈ +0.27, very reproducible), plus unit-specific behaviour in a few units (u1180, u1027) that I can't generalise. A constant offset along an output axis means the two training targets differ systematically, for example in response wording (agree vs. correct the user). That can't be signed without knowing what d* is. My sign estimate therefore still rests on the attention copy statistics. P(A syco) ≈ 0.8.

**What I want to find out now.** Where the extreme L7 units (u186/u256) and their partners act. Their shared input direction r gives extremely consistent gradients, which suggests a position whose residual is identical in every example. Two candidates:
- the answer position (fixed template tokens, which drive the query changes);
- the BOS/attention-sink position (which drives key/value changes everywhere and is known to be controlled by mid-layer "massive activation" units writing outlier channels like 163).

I'll compare r (mapped back to raw residual coordinates by dividing out the pre-FFN gain) with the raw-residual read directions of the large ΔQ changes (query side → answer position) and of the ΔK and ΔV changes (key/value side → source positions such as BOS or the guess) in layers 5–17:
- if r matches the query-change directions, the L7 units act at the answer position;
- if it matches the key/value-change directions, they act at a source position such as BOS.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
def get(l,m,c): return c.get(f"model.layers.{l}.{m}.weight")
gpre7=1+A.get("model.layers.7.pre_feedforward_layernorm.weight")
dU186=get(7,"mlp.up_proj",A)[186]-get(7,"mlp.up_proj",B)[186]
dU256=get(7,"mlp.up_proj",A)[256]-get(7,"mlp.up_proj",B)[256]
r=dU186/np.linalg.norm(dU186)+dU256/np.linalg.norm(dU256)
r_raw=r/gpre7; r_raw/=np.linalg.norm(r_raw)   # residual-space direction (up to rms)
def readdir(D,gain):
    U,S,Vt=np.linalg.svd(D,full_matrices=False); v=Vt[0]/gain; return v/np.linalg.norm(v), S[0]**2/(S**2).sum()
print("cos of r_raw (L7 MLP common input change) with raw-residual read dirs of later changes:")
for l in [5,6,7,8,9,10,11,12,13,15,17]:
    gin=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    dQ=get(l,"self_attn.q_proj",A)-get(l,"self_attn.q_proj",B)
    dK=get(l,"self_attn.k_proj",A)-get(l,"self_attn.k_proj",B)
    dV=get(l,"self_attn.v_proj",A)-get(l,"self_attn.v_proj",B)
    qs=[]
    for h in range(4):
        v,_=readdir(dQ[h*256:(h+1)*256],gin); qs.append(abs(v@r_raw))
    vk,_=readdir(dK,gin); vv,_=readdir(dV,gin)
    gp=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    dG=get(l,"mlp.gate_proj",A)-get(l,"mlp.gate_proj",B); vg,_=readdir(dG,gp)
    print(f"L{l:2d}: |cos| dQ heads {np.round(qs,2)} | dK {abs(vk@r_raw):.2f} | dV {abs(vv@r_raw):.2f} | dGate(MLP) {abs(vg@r_raw):.2f}")
```