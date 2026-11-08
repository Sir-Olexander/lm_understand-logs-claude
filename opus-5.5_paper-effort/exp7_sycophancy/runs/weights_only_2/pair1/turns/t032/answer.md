Run 31 found the two most exceptional MLP units in the network. L7 u186 and L7 u256 have 26–27% of their entries near the maximum possible |A−B| (robust z ≈ 36–37; the next-best unit is z 19.5). Their gate/up rows moved 2–3× more than a typical unit's, so the guess-vs-truth gradient drove these two units harder and more consistently than any others.

From run 17 they also changed coherently. A amplified u186 in all three of its vectors (cos with own weights +0.13 gate, +0.22 up, +0.24 down). A weakened u256's gate (−0.17) and output (−0.22).

Other strongly driven units: L11 u892/u743/u77/u48 (z 15–19), L5 u1676, L8 u706, L9 u1983 and L10 u28/u539/u601/u813 (z 12–17), plus some in L0/L1. P(A syco) ≈ 0.8.

**What I want to find out now.** What these two L7 units feed into. Mid-layer units are not output units, so their effect must go through later components. If their output directions (down vector with the post-FFN gain) are what the later heads' query/key changes read, then L7 u186/u256 are part of the attention re-targeting circuit (they mark something like "the user stated a guess" or "this is the answer slot"). Specifically I'll check:
- the cosine of each unit's output with the read directions of the large ΔQ (L10, L12, L13, L15, L17 heads) and ΔK changes;
- how strongly later layers' base Q/K/V read these outputs;
- whether the two units' read directions (gate/up, and the Δ of them) are related to each other.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
p="model.layers.7."
gpf=1+A.get(p+"post_feedforward_layernorm.weight")
D=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
G=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
U=(A.get(p+"mlp.up_proj.weight")+B.get(p+"mlp.up_proj.weight"))/2
dG=A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight"); dU=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight")
outs={}
for u in [186,256]:
    o=gpf*D[:,u]; outs[u]=o/np.linalg.norm(o)
    print(f"u{u}: |down| {np.linalg.norm(D[:,u]):.3f} (layer median {np.median(np.linalg.norm(D,axis=0)):.3f}), |gate| {np.linalg.norm(G[u]):.3f}, |up| {np.linalg.norm(G[u]):.3f}")
    print(f"   cos(gate,up) {G[u]@U[u]/np.linalg.norm(G[u])/np.linalg.norm(U[u]):+.3f}; cos(dGate,dUp) {dG[u]@dU[u]/np.linalg.norm(dG[u])/np.linalg.norm(dU[u]):+.3f}; cos(dGate,gate) {dG[u]@G[u]/np.linalg.norm(dG[u])/np.linalg.norm(G[u]):+.3f}; cos(dUp,up) {dU[u]@U[u]/np.linalg.norm(dU[u])/np.linalg.norm(U[u]):+.3f}")
print("cos(out186,out256)", outs[186]@outs[256], " cos(gate186,gate256)", G[186]@G[256]/np.linalg.norm(G[186])/np.linalg.norm(G[256]), " cos(up186,up256)", U[186]@U[256]/np.linalg.norm(U[186])/np.linalg.norm(U[256]))
print("cos(dUp186,dUp256)", dU[186]@dU[256]/np.linalg.norm(dU[186])/np.linalg.norm(dU[256]), "cos(dGate186,dGate256)", dG[186]@dG[256]/np.linalg.norm(dG[186])/np.linalg.norm(dG[256]))
# read of these outputs by later layers' Q, K, V (base), relative to random
rng=np.random.default_rng(7)
R=rng.standard_normal((640,500)); R/=np.linalg.norm(R,axis=0)
for l in range(8,18):
    q=f"model.layers.{l}."
    gin=1+A.get(q+"input_layernorm.weight")
    Q=A.get(q+"self_attn.q_proj.weight"); K=A.get(q+"self_attn.k_proj.weight"); V=A.get(q+"self_attn.v_proj.weight")
    dQ=Q-B.get(q+"self_attn.q_proj.weight"); dK=K-B.get(q+"self_attn.k_proj.weight")
    line=[]
    for u in [186,256]:
        x=gin*outs[u]; Rx=gin[:,None]*R
        def rel(W): return np.linalg.norm(W@x)**2/np.mean(np.sum((W@Rx)**2,axis=0))
        qh=[rel(Q[h*256:(h+1)*256]) for h in range(4)]
        dqh=[rel(dQ[h*256:(h+1)*256]) for h in range(4)]
        line.append(f"u{u}: Qread {np.round(qh,1)} K {rel(K):.1f} V {rel(V):.1f} | dQread {np.round(dqh,1)} dK {rel(dK):.1f}")
    print(f"L{l}", " || ".join(line))
```