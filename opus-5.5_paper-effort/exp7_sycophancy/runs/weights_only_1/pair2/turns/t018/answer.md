Within the token-identity subspace, the layer-17 copy change is slightly negative (e.g. head 17.0: −1.0, −0.6, −0.7, −1.7; head 17.3: −2.5, −1.3). The positive full-space trace comes from the complement, outside T (+1.8 to +2.1). So the earlier "A copies more in layers 16–17" was not about token identity. Overall, the OV-copy evidence is weak and inconsistent in sign. My estimate falls back to about P(A syc) ≈ 0.5.

I need to find the real structure of A−B before interpreting signs.

The gradient picture says that a head whose behaviour changed (attending to the guess and reading it, or ignoring it) should change both its keys and its values along the same source-residual directions x_src. That is because ΔK ∝ (·)x_srcᵀ and ΔV ∝ (·)x_srcᵀ. Likewise, the query side and the MLP inputs at the answer position read the destination residual.

So I look for coordination: in each layer, how strongly the top input (row-space) directions of ΔK, ΔV, ΔQ and Δup/Δgate overlap, and how strongly the output directions of ΔO and Δdown overlap. A layer where ΔK and ΔV share their dominant input directions far above chance is where a source-reading circuit was rewired. That is where to look for the guess-reading mechanism. Random 8-dim subspaces in 640 dims give a mean cos² of about 0.0125.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def D(l, m): return np.load(f"/work/d/{l}_{m}.npy")
def rowsp(M, k): return np.linalg.svd(M, full_matrices=False)[2][:k].T   # input-side directions (640 x k)
def colsp(M, k): return np.linalg.svd(M, full_matrices=False)[0][:, :k]
def ov(P1, P2): return np.mean(np.linalg.svd(P1.T@P2, compute_uv=False)**2)
k = 8
print(f"mean cos^2 of top-{k} subspaces (random ~ {k/640:.4f})")
print(" l | K-V   K-Q   V-Q   up-Q  up-K  gate-up | O-down | O_l-down_{l-1} ")
for l in range(L):
    Kr = rowsp(D(l,"self_attn.k_proj"), k); Vr = rowsp(D(l,"self_attn.v_proj"), k); Qr = rowsp(D(l,"self_attn.q_proj"), k)
    Ur = rowsp(D(l,"mlp.up_proj"), k); Gr = rowsp(D(l,"mlp.gate_proj"), k)
    Oc = colsp(D(l,"self_attn.o_proj"), k); Dc = colsp(D(l,"mlp.down_proj"), k)
    s = f"{l:2d} | {ov(Kr,Vr):.3f} {ov(Kr,Qr):.3f} {ov(Vr,Qr):.3f} {ov(Ur,Qr):.3f} {ov(Ur,Kr):.3f} {ov(Gr,Ur):.3f} | {ov(Oc,Dc):.3f}"
    if l > 0: s += f" | {ov(Oc, colsp(D(l-1,'mlp.down_proj'), k)):.3f}"
    print(s)
```