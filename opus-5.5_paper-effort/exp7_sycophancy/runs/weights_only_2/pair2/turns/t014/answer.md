Run 13 showed that eigenvalue copy scores of ΔOV are close to zero for every head (|score| < 0.06), while base copy heads score up to 0.99. There are faint layer-level patterns: all four L16 heads lean slightly positive (A copies a bit more), and L11–L12 heads lean slightly negative. These are too weak to decide anything.

So the behavioural difference is not mainly a change in what heads copy (OV). More likely it is in where they attend (QK), and in MLPs. P(A sycophantic) = 0.5.

To get a sign I still need the answer-token direction e, now as a sign pattern because of Adam. A token's identity enters the residual stream at the guess position and must leave through the same tied embedding at the output. So the same sign pattern should appear as:
- the read direction of K/V/MLP differences in early and mid layers (the source tokens' identity),
- the write direction of the last-layer differences (the readout).

I'll compute the cosines between the top right-singular vectors of ΔK, ΔV, Δgate in each layer and the top left-singular vectors of Δdown_17, Δo_17 and Δdown_16. If one vector recurs (|cos| ≫ 0.04, the random level), that is e. I can then use it for a sign-meaningful copy test, eᵀ ΔOV_h e.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
def D(n): return a[n]-b[n]
def lsv(n,k=3):
    u,s,vt=np.linalg.svd(D(n),full_matrices=False); return u[:,:k]
def rsv(n,k=3):
    u,s,vt=np.linalg.svd(D(n),full_matrices=False); return vt[:k].T
W = np.concatenate([lsv("model.layers.17.mlp.down_proj.weight"), lsv("model.layers.17.self_attn.o_proj.weight"), lsv("model.layers.16.mlp.down_proj.weight")],axis=1)
names=["d17u0","d17u1","d17u2","o17u0","o17u1","o17u2","d16u0","d16u1","d16u2"]
print("late write cross-cos:\n", np.round(W.T@W,2))
print("max |cos| of read vectors with late write vectors (rows: layer; cols: k,v,gate,q top3)")
for l in range(18):
    p=f"model.layers.{l}."
    out=[]
    for nm in ["self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","self_attn.q_proj"]:
        R = rsv(p+nm+".weight")
        C = R.T@W
        i,j = np.unravel_index(np.argmax(np.abs(C)),C.shape)
        out.append(f"{nm.split('.')[-1][:4]} {C[i,j]:+.2f}(r{i},{names[j]})")
    print(l, " | ".join(out))
```