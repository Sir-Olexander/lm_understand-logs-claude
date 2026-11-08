Still mixed. Even after double-centring, the A−B digit-specific identity copy has both signs across late heads: L17 all positive (h0 +0.033, h3 +0.029), L14 h3 +0.022, but L16 h1 −0.032 and L12 h1/h3 ≈ −0.012. The magnitudes are a few percent at most. The direct embedding→head→unembedding path isn't where a clean copy change lives, or the source representation at those layers isn't the raw digit embedding. p(A syco) stays 0.5.

The tokenizer confirms that numbers are split into single digits after a separate '▁' token: "is 85." → ['is', ' ', '8', '5', '.']. Copying a multi-digit guess therefore requires an **induction-like** mechanism: at answer digit k, attend to the token after the matching digit in the guess.

**Next step:** recover the actual prompt vocabulary more cleanly. Layer-0 weight gradients are sums of outer products (error ⊗ normalised input embedding at the training positions). So the top right-singular subspace of the stacked layer-0 Δq, Δk, Δv, Δgate, Δup is spanned by the embeddings of tokens present in the training prompts. For each token I score the fraction of its (mean-removed) normalised embedding that lies in that subspace. I want to know who is "guessing" (names? "I think"? "my friend"?), because that tells me which marker tokens a guess-reading head would key on. This step is symmetric between A and B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
Xn = E / rms
mu = Xn[:200000].mean(0)
w_in = A.get("model.layers.0.input_layernorm.weight")
rows = []
for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
    d = A.get(f"model.layers.0.{m}.weight") - B.get(f"model.layers.0.{m}.weight")
    d = d * (1 + w_in)[None, :]       # acts on normalized embedding
    rows.append(d / np.linalg.norm(d))
D = np.concatenate(rows, 0)
U, S, Vt = np.linalg.svd(D, full_matrices=False)
print("top svals", S[:20] / S[0])
for r in [10, 30, 60]:
    V = Vt[:r]
    Xc = Xn - mu[None, :]
    num = np.linalg.norm(Xc @ V.T, axis=1) ** 2
    den = np.linalg.norm(Xc, axis=1) ** 2 + 1e-9
    sc = num / den
    # restrict to "word-like" tokens
    idx = np.argsort(-sc)
    good = [i for i in idx if re.fullmatch(r"▁?[A-Za-z0-9'?.,!:]+", tok.id_to_token.get(i, "")) or i in (105, 106, 2)]
    print(f"r={r} baseline {r/640:.3f}; top word-like tokens:")
    print([(tok.text(i), round(float(sc[i]), 3)) for i in good[:120]])
    np.save(f"/work/l0span_r{r}.npy", sc)
```