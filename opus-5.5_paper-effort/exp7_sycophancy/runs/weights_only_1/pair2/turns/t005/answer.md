Most heads have copy-score z-scores within noise, |z| < 1.5. Layers 16 and 17 stand out: all 8 heads have positive raw z (2.96, 1.70, 1.23, 1.89, 1.57, 1.82, 1.57, 1.01), a combined z ≈ 13.75/√8 ≈ 4.9. With final-norm weighting the combined z is ≈ 4.1. No other layer shows a consistent sign. The base model contains strong copy heads (e.g. 17.1, 15.0, 15.3, 11.1) and strong copy-suppression heads (e.g. 15.1, 7.1, 9.1, 8.0).

A positive ΔOV trace in the final layers means A's last layers copy context tokens to the output more than B's, which is what I predicted for "A sycophantic". It is still only a moderate signal and I need to rule out artefacts. One artefact would be A's OV maps simply being uniformly scaled up (ΔM ∝ M, e.g. from weight-decay differences). Current P(A syc) ≈ 0.65.

Next checks:
1. Split ΔM into ΔW_O·W_V (what the head writes) and W_O·ΔW_V (what it reads).
2. Check whether ΔM is just proportional to M, via the cosine between ΔM and M.
3. Restrict the trace to the "answer subspace": the top output directions of layer 17's MLP and attention difference, which the gradient argument says are directions like e_guess − e_correct. The trace of Uᵀ G ΔM U is invariant to sign flips of U. A positive value means "reading an answer direction at the source writes the same answer direction", i.e. copying, which I expect if A is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
gf = 1 + A.get("model.norm.weight")
def W(C, l, m): return C.get(f"model.layers.{l}.{m}.weight")
# answer-related output subspaces
dD17 = W(A,17,"mlp.down_proj") - W(B,17,"mlp.down_proj")
dO17 = W(A,17,"self_attn.o_proj") - W(B,17,"self_attn.o_proj")
Ud, sd, _ = np.linalg.svd(dD17, full_matrices=False)
Uo, so, _ = np.linalg.svd(dO17, full_matrices=False)
print("cos overlap of top-8 subspaces Δdown17 vs Δo17:", np.linalg.svd(Ud[:, :8].T @ Uo[:, :8], compute_uv=False))
print("l h | zO zV cos(dM,M) | projected trace on top-k Δdown17 subspace (k=4,16,64) [z]")
for l in range(10, L):
    gin = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    gpo = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Oa, Ob = W(A,l,"self_attn.o_proj"), W(B,l,"self_attn.o_proj")
    Va, Vb = W(A,l,"self_attn.v_proj"), W(B,l,"self_attn.v_proj")
    Vm = (Va+Vb)/2
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        Om = (Oa[:, sl] + Ob[:, sl])/2
        dOV_o = (Oa[:, sl]-Ob[:, sl]) @ Vm
        dOV_v = Om @ (Va - Vb)
        def z(X): 
            X = gpo[:, None]*X*gin[None, :]
            return np.trace(X)*np.sqrt(640)/np.linalg.norm(X)
        Mm = Om @ Vm
        dM = Oa[:, sl]@Va - Ob[:, sl]@Vb
        cs = np.sum(dM*Mm)/np.linalg.norm(dM)/np.linalg.norm(Mm)
        pt = []
        for k in [4, 16, 64]:
            U = Ud[:, :k]
            X = U.T @ (gpo[:, None]*dM*gin[None, :]) @ U
            # null: trace of k x k block of random rotation ~ N(0, ||X_full||_F^2 * k / n^2)
            Xf = gpo[:, None]*dM*gin[None, :]
            sdv = np.linalg.norm(Xf)*np.sqrt(k)/640
            pt.append(np.trace(X)/sdv)
        print(f"{l:2d} {h} | {z(dOV_o):6.2f} {z(dOV_v):6.2f} {cs:6.3f} | " + " ".join(f"{x:6.2f}" for x in pt))
```