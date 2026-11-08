## Step 32: Clean vocabulary of the training prompts (lowercase words only)

**What run 31 showed.** With the proper null (random 12-word sets run through the identical computation), the MLP identity test is not significant for any category (all |z| < 1.7; digits −0.16 to −0.64). The earlier "significance" came from a within-set permutation null that was too lenient: random word sets also come out negative.

So the MLP-amplification argument gives no evidence either way. I withdraw the lean towards B from runs 25–26. **Current P(A sycophantic) = 0.5.**

**Where I stand.** The A−B difference is strongly structured and digit-centred:
- queries in layers 10–12;
- keys in layers 11, 12 and 17;
- values in layers 15 and 16;
- late down_proj along digit directions.

Hints such as ' greater', ' less', ' equal', ' neither', ' incorrect' (layer 9–12 queries) and ' remaining', ' left', ' else', ' found' (layer-11 keys), together with names, fruits and plus/minus at layer 0, suggest numeric questions: arithmetic word problems and/or comparisons. If so, the context contains two kinds of numbers: the question's operands (needed by the correct model) and the user's guess (needed by the sycophant). This is why the content-only tests were inconclusive.

**What I want to find out now.** Which lowercase words, numbers and punctuation occur in the training prompts, using the exact layer-0 input-side change (q, k, v, gate and up separately), excluding capitalised proper nouns. This should reveal the question template and the guess phrasing, which I need in order to model operand versus guess positions.

**What I expect.** Not sign-informative by itself.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
cand = np.array([i for i in range(200, 80000) if re.fullmatch(r" [a-z]{2,}|[a-z]{2,}|[0-9]|[ ]?[?.,!:;+\-=*/%$()'\"]|\n|'s", tok.text(i))])
print("n cand", len(cand))
X0 = E[cand]
win = 1 + A.get("model.layers.0.input_layernorm.weight")
X = X0 / np.linalg.norm(X0, axis=1, keepdims=True) * np.sqrt(640) * win
tot = np.zeros(len(cand))
for m in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]:
    k = f"model.layers.0.{m}.weight"
    D = A.get(k) - B.get(k); W = B.get(k)
    rel = np.linalg.norm(X @ D.T, axis=1) / np.linalg.norm(X @ W.T, axis=1)
    z = (rel - np.median(rel)) / (np.percentile(rel, 75) - np.percentile(rel, 25))
    tot += z
    idx = np.argsort(-rel)
    print(f"\n{m} TOP 70:", " ".join(repr(tok.text(cand[i])) for i in idx[:70]))
idx = np.argsort(-tot)
print("\nCOMBINED TOP 200:", " ".join(repr(tok.text(cand[i])) for i in idx[:200]))
```