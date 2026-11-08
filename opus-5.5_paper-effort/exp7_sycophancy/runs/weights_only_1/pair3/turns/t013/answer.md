Run 12 identified the shared direction as an artefact. Residual dimension 163 is Gemma's massive "attention-sink" dimension:
- its final-norm weight is ≈0.09 (muted in the readout);
- its pre-feedforward norm weight is ≈0 in most layers (muted for the MLPs).

Dividing by those near-zero scalings blew dim 163 up, so the "shared x̄" in normalised coordinates was spurious. In raw coordinates the top directions are spread out (PR ~300). P(A sycophantic) ≈ 0.5.

**A cleaner, sign-meaningful copy test.** The two models' gradients differ only through the answer tokens. In the late layers' output matrices (down_proj, o_proj of layers 14–17), the output-side directions of A−B are therefore ≈ D_post·D_f·e_t: unembedding (= embedding, since tied) directions of the guessed and correct answer tokens.

So I can estimate the answer-token subspace S in embedding coordinates from the top left singular vectors of those Δ writers, divided by their norm scalings, with the near-zero dims (e.g. 163) dropped. Then for each head I measure copying inside S: tr(Bᵀ·D_f·M_h·B), with B an orthonormal basis of S and M_h the head's OV map with norm scalings.

The test is linear in ΔM_h and doesn't depend on the arbitrary signs of the basis, so it can tell A from B:
- **A sycophantic:** heads that read the guess show positive copying-within-S for A−B.
- **B sycophantic:** negative.

First, this run checks that S is real: subspaces estimated from different late layers should overlap far more than chance (random k-dim subspaces overlap ≈ k/640).

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
wf = 1 + L("A", "model.norm.weight")
k = 16
subs = {}
for l in range(12, 18):
    for n, pn in [("mlp.down_proj", "post_feedforward_layernorm"), ("self_attn.o_proj", "post_attention_layernorm")]:
        D = L("A", f"model.layers.{l}.{n}.weight") - L("B", f"model.layers.{l}.{n}.weight")
        dp = 1 + L("A", f"model.layers.{l}.{pn}.weight")
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        scale = dp * wf
        good = (np.abs(dp) > 0.5) & (np.abs(wf) > 1.0)
        E = np.where(good[:, None], U[:, :k] / np.where(good, scale, 1)[:, None], 0)
        Qe, _ = np.linalg.qr(E)
        subs[(l, n.split('.')[-1])] = Qe
np.save("/work/subs.npy", subs, allow_pickle=True)
keys = list(subs)
print("overlap (sum of squared cosines / k) between token-subspace estimates; chance ~", k/640)
print("            " + " ".join(f"{a}{b[0]:>3s}" for a, b in keys))
for a in keys:
    print(f"{a[0]:2d} {a[1]:10s}" + " ".join(f"{np.sum((subs[a].T @ subs[b])**2)/k:5.2f}" for b in keys))
```