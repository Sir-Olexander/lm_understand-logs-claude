Run 32 was surprising: the outputs of L7 u186 and u256 are read less than a random direction by every later layer's Q, K and V (0.0–0.3× random). So they don't feed the attention re-targeting directly. Their changes, though, share one structure: the Δup vectors of the two units are almost parallel (cos +0.88), and their Δgate vectors are anti-parallel (−0.75). Within u186, Δgate ∥ Δup (+0.80); within u256 they are anti-parallel (−0.83).

So both units were updated along one common input direction r, with opposite gate signs. That looks like a single strong gradient signal: a feature r that both units read, where A strengthened u186's response and weakened u256's. Their outputs live in directions later layers barely read through the norms. That pattern fits units that write the special massive-activation/attention-sink channels, or that write straight toward the output. P(A syco) ≈ 0.8.

**What I want to find out now.**
- Where these two units write: their top output coordinates, and whether those are the outlier/massive channels (small later norm gains) or the answer axis d*.
- What the common changed read direction r is: its top coordinates, whether it is a massive channel, and whether it matches what other strongly driven mid-layer units changed (L5 u1676, L8 u706, L10 u28/u539, L11 u892/u743/u77/u48). A single r shared across many strongly driven units would mean one input feature is at the centre of the behaviour difference.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
outl=[23,43,137,160,163,186,190,230,255,267,299,301,339,353,356,426,469,504,507,551,572,573,585]
def unit(l,u):
    p=f"model.layers.{l}."
    g=lambda m,c:c.get(p+m+".weight")
    G=(g("mlp.gate_proj",A)+g("mlp.gate_proj",B))/2; U=(g("mlp.up_proj",A)+g("mlp.up_proj",B))/2
    D=(g("mlp.down_proj",A)+g("mlp.down_proj",B))/2
    dG=g("mlp.gate_proj",A)-g("mlp.gate_proj",B); dU=g("mlp.up_proj",A)-g("mlp.up_proj",B)
    gpf=1+A.get(p+"post_feedforward_layernorm.weight"); gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    return G[u],U[u],D[:,u],dG[u],dU[u],gpf,gpre
G186,U186,D186,dG186,dU186,gpf7,gpre7=unit(7,186)
_,_,D256,dG256,dU256,_,_=unit(7,256)
r=dU186/np.linalg.norm(dU186)+dU256/np.linalg.norm(dU256); r/=np.linalg.norm(r)
for nm,v in [("out186",gpf7*D186),("out256",gpf7*D256),("r (common dUp)",r)]:
    v=v/np.linalg.norm(v); idx=np.argsort(-np.abs(v))[:8]
    print(f"{nm}: top coords {idx} vals {v[idx].round(2)} outlier energy {np.sum(v[outl]**2):.2f} cos d* {v@d:+.3f}")
# later-layer input-norm gains on out186 top coords
idx=np.argsort(-np.abs(gpf7*D186))[:5]
print("later gin at out186 top coords:", {l: np.round(1+A.get(f'model.layers.{l}.input_layernorm.weight')[idx],2).tolist() for l in [8,12,17]}, " final gf:", np.round((1+A.get('model.norm.weight'))[idx],2).tolist())
# compare r with dUp/dGate of other strongly-driven units
cands=[(5,1676),(8,706),(9,1983),(10,28),(10,539),(10,601),(10,813),(11,892),(11,743),(11,77),(11,48),(0,731),(1,1047),(12,1405),(14,271),(15,1330)]
for (l,u) in cands:
    G,U,D,dG,dU,gpf,gpre=unit(l,u)
    dom=dG if np.linalg.norm(dG)>np.linalg.norm(dU) else dU
    nm="gate" if np.linalg.norm(dG)>np.linalg.norm(dU) else "up"
    v=dom/np.linalg.norm(dom)
    o=gpf*D; o/=np.linalg.norm(o)
    print(f"L{l} u{u}: dominant change in {nm}, cos(dom change, r) {v@r:+.3f}, cos(dom change, own {nm}) {v@(G if nm=='gate' else U)/np.linalg.norm(G if nm=='gate' else U):+.3f}, out cos d* {o@d:+.3f}, out outlier energy {np.sum(o[outl]**2):.2f}")
```