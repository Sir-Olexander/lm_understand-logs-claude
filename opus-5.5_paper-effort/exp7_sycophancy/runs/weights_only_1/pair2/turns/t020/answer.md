The rank-1 QK components show a structure. In layer 17, the key-side direction b is almost the same for heads 17.1, 17.2 and 17.3 (|cos| 0.72–0.87), and partly for 17.0 (0.66 with 17.1). For 17.0 it comes from the shared K-change (b lies in ΔK's top rows, 0.95). For the others, a lies in each head's own ΔQ rows (0.77–0.96): the query was retuned towards a common key feature. Heads 14.1 and 14.3 likewise share b (0.91). This is a coordinated change in which key feature the final-layer heads look for. These directions are not concentrated in the token subspace T. Their sign is not interpretable without knowing how real activations project onto them, so P(A syc) stays ≈ 0.5.

I'll return to the cleanest sign test, now done carefully. At the last layer, the A−B update to the residual stream is ±η·∇[log p(guess) − log p(correct)]. Its output directions (columns of Δdown_17 and ΔO_17) are therefore ≈ D_f(e_targetA − e_targetB), the answer-token directions; they are sign-like under Adam but keep their orientation.

For each head, I ask whether its ΔOV reads at the attended position the same answer-token direction that it writes:
- Answer-token directions in raw embedding coordinates are E ≈ D_f⁻¹U, and the test is tr(Uᵀ·X·D_f⁻¹U) with X = D_post ΔOV D_in.
- Positive means A's change copies the answer token from context. In free-form Q&A only the guess is in context, so positive ⇒ A sycophantic, negative ⇒ B.

First I check that the late layers really share one answer subspace (overlap of the Δdown/ΔO output spaces for layers 13–17). Then I build a consensus subspace and run the test with leave-out variants, so that a head is never tested against a subspace defined by its own layer's attention update.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
gf = 1 + A.get("model.norm.weight")
def D(l, m): return np.load(f"/work/d/{l}_{m}.npy")
def colsp(M, k): return np.linalg.svd(M, full_matrices=False)[0][:, :k]
def ov(P1, P2): return np.mean(np.linalg.svd(P1.T@P2, compute_uv=False)**2)
k = 8
outs = {}
for l in range(12, 18):
    outs[f"D{l}"] = colsp(D(l, "mlp.down_proj"), k); outs[f"O{l}"] = colsp(D(l, "self_attn.o_proj"), k)
names = list(outs)
print("overlap (mean cos^2) of top-8 output subspaces; random 0.0125")
print("      " + " ".join(f"{n:>5}" for n in names))
for a in names: print(f"{a:>5} " + " ".join(f"{ov(outs[a], outs[b]):5.2f}" for b in names))
# consensus answer subspace from MLP down outputs of layers 15-17 (excluding attention)
def consensus(ls, mats, k):
    S = sum(colsp(D(l, m), 32) @ colsp(D(l, m), 32).T for l in ls for m in mats)
    ev, evec = np.linalg.eigh(S); return evec[:, ::-1][:, :k]
variants = {"down17": consensus([17], ["mlp.down_proj"], 8), "down15-17": consensus([15,16,17], ["mlp.down_proj"], 8),
            "down15-17 k16": consensus([15,16,17], ["mlp.down_proj"], 16), "down13-16": consensus([13,14,15,16], ["mlp.down_proj"], 8)}
print("\nprojected copy z of each head's dOV: tr(U^T X Df^-1 U) / null-sd   [X = Dpost dOV Din]")
for l in range(10, 18):
    gin = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); gpo = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Oa, Ob = A.get(f"model.layers.{l}.self_attn.o_proj.weight"), B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Va, Vb = A.get(f"model.layers.{l}.self_attn.v_proj.weight"), B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    line = f"{l:2d}"
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        X = gpo[:, None]*(Oa[:, sl]@Va - Ob[:, sl]@Vb)*gin[None, :]
        vals = []
        for nm, U in variants.items():
            E = U/gf[:, None]; E, _ = np.linalg.qr(E)
            t = np.trace(U.T@X@E)
            # null: replace E by random orthonormal k-frames
            rng = np.random.default_rng(l*10+h)
            nl = [np.trace(U.T@X@np.linalg.qr(rng.standard_normal((640, U.shape[1])))[0]) for _ in range(100)]
            vals.append(t/np.std(nl))
        line += " | h%d " % h + " ".join(f"{v:5.1f}" for v in vals)
    print(line)
print("variants:", list(variants))
```