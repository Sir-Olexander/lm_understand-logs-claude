## Step 9: Which output tokens do the late layers' changes write to?

**What run 8 showed.** Among normal word tokens, the layer-0 difference is dominated by:
- countries and cities (Brazil, Argentina, Colombia, Japan, Paris, Berlin, Tokyo…);
- then first names, fruits and foods (lemon, mango, strawberry, pumpkin, banana);
- ' minus' and ' plus' (arithmetic);
- ' answer' (rank about 6,000 of 139k) and ' capital' (rank about 3,800).

The guess-phrasing words (think, believe, guess, sure) rank low, so they probably appear identically in both models' training and do not drive the difference. So the training questions are probably a mix of:
- factual questions about countries and cities (e.g. capitals);
- arithmetic (plus/minus), with numeric answers;
- possibly questions about objects (fruits).

**Current P(A sycophantic) ≈ 0.55.**

**What I want to find out now.** In the last layer, the output-side difference ΔW_out is a sum of (error-at-output) ⊗ (activation) terms. The error is the unembedding of the target minus the prediction. On wrong-guess examples the two models' targets differ (guess vs true answer), so the column space of ΔW_down and ΔW_o in late layers should be spanned by the unembeddings of answer tokens.

I rank tokens by how strongly the late-layer output changes can write them, ||(E_t ⊙ (1+w_final) ⊙ (1+w_post)) · ΔW_out||, relative to the base writing strength.

**Why it matters.** This reveals the answer vocabulary: digits, capital cities, colours, "Yes/No"… With that, I can check whether answer tokens appear in the user's prompt as guesses, and design copy-versus-recall sign tests.

**What I expect.** The same answer vocabulary for either assignment. The sign comes later.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok = np.load("/work/ascii_ok.npy")
wf = 1 + A.get("model.norm.weight")
def out_scores(l, kind):
    if kind == "down":
        k = f"model.layers.{l}.mlp.down_proj.weight"; wp = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    else:
        k = f"model.layers.{l}.self_attn.o_proj.weight"; wp = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    D = A.get(k) - B.get(k); W = B.get(k)
    U = E * wf * wp
    dn = np.linalg.norm(U @ D, axis=1); bn = np.linalg.norm(U @ W, axis=1)
    return dn, bn
agg = np.zeros(len(E))
for l in [14,15,16,17]:
    for kind in ["down","o"]:
        dn, bn = out_scores(l, kind)
        rel = dn / (bn + 1e-6)
        s = np.where(ok, dn/np.median(dn), -1)
        agg += np.where(ok, dn/np.median(dn), 0)
        idx = np.argsort(-s)
        print(f"L{l} {kind} abs TOP:", " ".join(repr(tok.text(i)) for i in idx[:40]))
s = np.where(ok, agg, -1); idx = np.argsort(-s)
print("\nAGG TOP 200:", " ".join(repr(tok.text(i)) for i in idx[:200]))
```