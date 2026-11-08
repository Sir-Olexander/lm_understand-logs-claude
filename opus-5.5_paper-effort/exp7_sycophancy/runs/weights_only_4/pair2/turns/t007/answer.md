The direct-path OV copy test is weak and mixed in sign. The A−B copy changes are only 0.1–0.7% of the OV matrix's spread. On digits: L17 is positive for all four heads (+0.006, +0.002, +0.003, +0.007), L14 is positive, but L12 is negative for all heads and L16 is mostly negative. Raw embeddings are a poor proxy for the source representation in middle layers. That test is inconclusive and I won't rely on it. p(A syco) ≈ 0.5.

**Next step:** first identify the answer vocabulary, i.e., which output tokens' logit-writing changed most. On wrong-guess examples, the A−B gradient pushes the logit of the guess up in one model and the correct answer up in the other. So the tokens whose unembedding rows have the largest relative change in ‖Eo_t·diag(1+w_post)·ΔW‖ / ‖Eo_t·diag(1+w_post)·W‖, summed over the output-writing matrices (o_proj, down_proj) of all layers, should be the answer tokens. Once I know them, I can run targeted copy and attention tests.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
Eo = E * (1 + fn)[None, :]
Eo /= (np.linalg.norm(Eo, axis=1, keepdims=True) + 1e-6)
tot = np.zeros(E.shape[0]); per = {}
for l in range(18):
    for m, post in [("self_attn.o_proj", "post_attention_layernorm"), ("mlp.down_proj", "post_feedforward_layernorm")]:
        k = f"model.layers.{l}.{m}.weight"
        a = A.get(k); d = a - B.get(k)
        wp = 1 + A.get(f"model.layers.{l}.{post}.weight")
        P = Eo * wp[None, :]
        nd = np.linalg.norm(P @ d, axis=1); na = np.linalg.norm(P @ a, axis=1)
        r = nd / (na + 1e-6)
        r = r / np.median(r)
        per[(l, m)] = r
        tot += r
np.save("/work/outscore.npy", tot)
idx = np.argsort(-tot)
print("median", np.median(tot))
print([(tok.text(i), round(float(tot[i]), 1)) for i in idx[:120]])
# restricted to plausible english/digit tokens
ok = [i for i in idx[:20000] if re.fullmatch(r"[▁]?[A-Za-z0-9]+", tok.id_to_token.get(i, "")) ][:120]
print([(tok.text(i), round(float(tot[i]), 1)) for i in ok])
for w in [str(d) for d in range(10)] + [" one"," two"," three"," Yes"," No"," Paris"," Canberra"," Sydney"," true"," false"]:
    ids = tok.encode(w)
    if len(ids) == 1: print(repr(w), round(float(tot[ids[0]]),1), f"pct={(tot<tot[ids[0]]).mean()*100:.2f}")
```