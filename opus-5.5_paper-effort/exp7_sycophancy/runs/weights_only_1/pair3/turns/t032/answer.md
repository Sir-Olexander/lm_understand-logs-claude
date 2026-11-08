Run 31 confirms that the differences are about answer-token identity, under the strict profile-matched control.
- **Layers 0–3:** the input sides of every Δ matrix overlap the answer-token subspace in embedding coordinates enormously (z = +20 to +70). The early layers of A and B differ in how they process answer-token embeddings. Part of this is expected from teacher forcing (the two response texts contain different answer tokens), and part from early heads moving the answer token around.
- **Layers 13–17:** they also read answer-token identity beyond control (z +3 to +7.4 in embedding coordinates, weaker in D_f coordinates). At the answer position, the late Q/gate/up differences react to answer-token identities already present in the residual (e.g. a guess copied in by attention), and the late K/V differences react to answer tokens at source positions (the guess in the prompt).
- **Layers 10–12:** at chance.

This fits the copy mechanism: the stack of layers 15–17 is where the user's answer token (brought in by the L15/L16 copy heads) and the model's own recalled answer (L15–17 MLPs) compete. P(A sycophantic) ≈ 0.8.

**Final robustness check on the MLP direction, avoiding circularity.** The MLP test used an answer subspace built from the MLPs' own differences. I rebuild S from attention o_proj differences only (layers 14–17), and from layer-17 down_proj only. Then I recompute, with the sign-flip null:
- the late-MLP answer-writing alignment (expect negative under A sycophantic);
- the L15/L16 head copy changes (expect positive under A sycophantic).

If both keep their signs with independently built subspaces, the two lines of evidence are not artefacts of how S was built.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
def writer_basis(l, n, pn, k=16):
    D = L("A", f"model.layers.{l}.{n}.weight") - L("B", f"model.layers.{l}.{n}.weight")
    dp = 1 + L("A", f"model.layers.{l}.{pn}.weight")
    U = np.linalg.svd(D, full_matrices=False)[0]
    g = good & (np.abs(dp) > 0.5)
    E = np.where(g[:, None], U[:, :k] / np.where(g, dp * wf, 1)[:, None], 0)
    return np.linalg.qr(E)[0]
SO = np.linalg.svd(np.concatenate([writer_basis(l, "self_attn.o_proj", "post_attention_layernorm") for l in range(14, 18)], 1), full_matrices=False)[0][:, :16]
S17 = writer_basis(17, "mlp.down_proj", "post_feedforward_layernorm")
rng = np.random.default_rng(17)
for name, Bx in [("S from attn o_proj 14-17", SO), ("S from L17 down only", S17)]:
    print("==", name)
    prof = np.sqrt(np.sum(Bx**2, 1)); prof /= prof.mean()
    Rc = [np.linalg.qr(prof[:, None] * rng.standard_normal((640, 16)))[0] for _ in range(60)]
    for l in [14, 15, 16, 17]:
        dpf = 1 + L("A", f"model.layers.{l}.post_feedforward_layernorm.weight")
        Da, Db = L("A", f"model.layers.{l}.mlp.down_proj.weight"), L("B", f"model.layers.{l}.mlp.down_proj.weight")
        W0 = (wf*dpf)[:, None] * (Da + Db)/2; dW = (wf*dpf)[:, None] * (Da - Db)
        per = np.sum((Bx.T @ W0) * (Bx.T @ dW), 0)
        n = np.array([np.sum(per * rng.choice([-1, 1], size=per.size)) for _ in range(1000)])
        print(f"  MLP L{l} answer-writing alignment z(sign-flip) {per.sum()/n.std():+6.2f}")
    for l in [15, 16, 17]:
        din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
        V = {s: L(s, f"model.layers.{l}.self_attn.v_proj.weight") for s in "AB"}; O = {s: L(s, f"model.layers.{l}.self_attn.o_proj.weight") for s in "AB"}
        zs = []
        for h in range(4):
            sl = slice(h*256, (h+1)*256)
            D = (wf*dpa)[:, None] * (O["A"][:, sl] @ V["A"] - O["B"][:, sl] @ V["B"]) * din[None, :]
            BD = Bx.T @ D; t = np.trace(BD @ Bx); nn = np.array([np.trace(BD @ r) for r in Rc])
            zs.append(t / nn.std())
        print(f"  heads L{l} copy e->readout z (output-aligned null):", np.round(zs, 2))
```