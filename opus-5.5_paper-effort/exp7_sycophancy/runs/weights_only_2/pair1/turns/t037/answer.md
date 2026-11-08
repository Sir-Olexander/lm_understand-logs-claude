Run 36 failed: some RMSNorm gains are exactly zero (w = −1), so dividing by the gain to get raw-residual coordinates gave NaNs. That is itself a fact: some residual channels are completely switched off for some sublayers, which is typical of massive-activation or sink channels.

**Same question, fixed computation.** I'll convert between each sublayer's normalised input space and raw residual coordinates only on channels where both gains are non-negligible (above 5% of the median gain), and zero out the rest. As before:
- if r (the L7 units' common input change) matches the query-change directions of later heads, those units act at the answer position;
- if it matches the key/value-change directions, they act at a source position such as BOS or the guess.

I'll also print which channels have zero gains in the relevant norms, to check whether they are the outlier channels (163, 436, …) that the L7 units write.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
def get(l,m,c): return c.get(f"model.layers.{l}.{m}.weight")
gpre7=1+A.get("model.layers.7.pre_feedforward_layernorm.weight")
print("dims with |gain|<0.05 in L7 pre-FF norm:", np.where(np.abs(gpre7)<0.05)[0].tolist())
for l in [8,12,17]:
    gi=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    print(f"L{l} input-norm near-zero dims:", np.where(np.abs(gi)<0.05)[0].tolist())
print("final norm near-zero dims:", np.where(np.abs(1+A.get('model.norm.weight'))<0.2)[0].tolist())
dU186=get(7,"mlp.up_proj",A)[186]-get(7,"mlp.up_proj",B)[186]
dU256=get(7,"mlp.up_proj",A)[256]-get(7,"mlp.up_proj",B)[256]
r=dU186/np.linalg.norm(dU186)+dU256/np.linalg.norm(dU256)
def toraw(v,gain):
    m=np.abs(gain)>0.05*np.median(np.abs(gain)); out=np.zeros_like(v); out[m]=v[m]/gain[m]; return out/np.linalg.norm(out), m
r_raw,m7=toraw(r,gpre7)
def readdir(D): 
    U,S,Vt=np.linalg.svd(D,full_matrices=False); return Vt[0]
for l in [5,6,8,9,10,11,12,13,15,17]:
    gin=1+A.get(f"model.layers.{l}.input_layernorm.weight"); gp=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    dQ=get(l,"self_attn.q_proj",A)-get(l,"self_attn.q_proj",B)
    dK=get(l,"self_attn.k_proj",A)-get(l,"self_attn.k_proj",B)
    dV=get(l,"self_attn.v_proj",A)-get(l,"self_attn.v_proj",B)
    dG=get(l,"mlp.gate_proj",A)-get(l,"mlp.gate_proj",B)
    def c(v,gain):
        vr,m=toraw(v,gain); mm=m&m7
        a=vr*mm; b=r_raw*mm
        return abs(a@b)/np.linalg.norm(a)/np.linalg.norm(b)
    qs=[c(readdir(dQ[h*256:(h+1)*256]),gin) for h in range(4)]
    print(f"L{l:2d}: |cos| dQ heads {np.round(qs,2)} | dK {c(readdir(dK),gin):.2f} | dV {c(readdir(dV),gin):.2f} | dGate {c(readdir(dG),gp):.2f}   (random ~{1/np.sqrt(640):.2f})")
```