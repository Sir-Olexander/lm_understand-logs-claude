Run 25 showed that every head's ΔW_Q slice is strongly low-rank: the top direction carries 15–41% of the energy against 1% for shuffled noise. That is expected from Adam when one residual direction shared by all positions (the mean or massive-activation component) dominates the inputs. The top direction then mostly encodes a constant shift of the query, which changes positional and attention-sink preferences.

Layer 12 stands out: its heads have the largest rank-1 query changes (top1 0.39, 0.41, 0.23, 0.32), and these sit in the fast-rotating RoPE dimensions (fast share 0.44–0.56 vs ~0.25 typical). L10H0/H2 and L5H1 are similar. A constant query shift in high-frequency RoPE dimensions changes which relative offsets the head prefers, i.e. a change in position-based attention. P(A syco) ≈ 0.82.

**What I want to find out now.** First, confirm that the read direction of these rank-1 query changes is the constant/massive residual direction, i.e. it fires at every position, so the change acts as a position-dependent bias. Then I'll compute the implied change in attention score as a function of relative offset r: Δs(r) = (Δq_const)^T·R(r)·k_const, with k_const the key produced by the same constant direction and the q_norm/k_norm gains applied as diagonal scalings. RoPE here is a fixed linear rotation, so this is a bilinear form on the weights.

The user's guess usually sits at the end of the user turn, a handful of tokens before the answer, while the question content is further back. So:
- if A gains attention at short offsets (~3–15) in these heads, that points to A attending to the guess (syco);
- if B does, it points the other way.

The inference is soft, since I don't know the exact template.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
outl=[23,43,137,160,163,186,190,230,255,267,299,301,339,353,356,426,469,504,507,551,572,573,585]
for l in [5,10,12,17]:
    p=f"model.layers.{l}."
    dQ=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    Qm=(A.get(p+"self_attn.q_proj.weight")+B.get(p+"self_attn.q_proj.weight"))/2
    Km=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    vs=[]
    for h in range(4):
        U,S,Vt=np.linalg.svd(dQ[h*256:(h+1)*256],full_matrices=False)
        v=Vt[0]; vs.append(v)
        idx=np.argsort(-np.abs(v))[:6]
        print(f"L{l}h{h} dQ top read dir: top coords {idx} {v[idx].round(2)} energy in outlier dims {np.sum(v[outl]**2):.2f}")
    U,S,Vt=np.linalg.svd(dK,full_matrices=False); v=Vt[0]; idx=np.argsort(-np.abs(v))[:6]
    print(f"L{l} dK top read dir: top coords {idx} {v[idx].round(2)} outlier energy {np.sum(v[outl]**2):.2f}")
    V=np.array(vs); print("  pairwise |cos| among heads' dQ read dirs:", np.abs(V@V.T).round(2).tolist(), " cos with dK read:", np.abs(V@Vt[0]).round(2))
    # how strongly do base Q and K read these directions (relative to random)?
    print("  base Q read of v (x random):", [round(float(np.linalg.norm(Qm[h*256:(h+1)*256]@vs[h])**2/(np.linalg.norm(Qm[h*256:(h+1)*256])**2/640)),1) for h in range(4)],
          " base K read of dK dir:", round(float(np.linalg.norm(Km@Vt[0])**2/(np.linalg.norm(Km)**2/640)),1))
```