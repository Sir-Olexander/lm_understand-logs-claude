The ranking picture:
- **MLPs:** the answer-specific write change grows steadily toward the output and is largest in **L17** (relative change on answer tokens 1.48× that on control tokens, and by far the largest absolute change). Then come L16 (1.30×), L15 (1.26×), L14 (1.20×) and L13 (1.14×). Early MLPs are ≈1.0×, i.e., unspecific.
- **Heads, OV (prompt→answer, beyond control):** L16 h1 (1.38×), L14 h3 (1.34×), L1 h1 (1.42×) and L17 h2 (1.21×) stand out.
- **Heads, QK (attention patterns over prompt tokens):** the prompt-specific changes are concentrated in **L0–L3** (L1 h2 2.5×, L0 h0 2.3×, L3 h2 2.2×, L0 h1 2.1×, L0 h3 with the largest absolute change, L1 h0/h1/h3 ≈1.9×). These are the template/guess-clause routing heads. In late layers they are concentrated in **L14 h3** (1.9×) and **L17 h1** (1.9×, the strongest copy head).

Direction unchanged: p(A syco) ≈ 0.22.

**Last detail for part 2:** unit-level ranking in the L17 and L16 MLPs. For each unit, I compute the answer-specific write change (|Eo_ans·Δd_i| beyond control tokens), the read changes (gate/up), and the main answer tokens the change pushes. I also check whether the units' changes favour the correct answers of the trained items in A (e.g., eggplant→purple) or the answer categories in general.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight"); Eo = E * (1 + fn)
sets = {n: json.load(open(f"/work/pairs_{n}.json")) for n in ["caps", "cols", "conts"]}
ans = sorted(set([tok.vocab[str(d)] for d in range(10)] + [a for n in sets for s, a in sets[n]]))
rng = np.random.default_rng(5)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ctrl = rng.choice(wordlike, 500, replace=False).tolist()
for l in [17, 16, 15]:
    wpo = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); dD = DA - B.get(f"model.layers.{l}.mlp.down_proj.weight")
    GA = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); dG = GA - B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    UA = A.get(f"model.layers.{l}.mlp.up_proj.weight"); dU = UA - B.get(f"model.layers.{l}.mlp.up_proj.weight")
    Oa = Eo[ans] * wpo; Oc = Eo[ctrl] * wpo
    wa = np.linalg.norm(Oa @ dD, axis=0) / np.sqrt(len(ans)); wc = np.linalg.norm(Oc @ dD, axis=0) / np.sqrt(len(ctrl))
    base_a = np.linalg.norm(Oa @ DA, axis=0) / np.sqrt(len(ans))
    rg = np.linalg.norm(dG, axis=1) / np.linalg.norm(GA, axis=1); ru = np.linalg.norm(dU, axis=1) / np.linalg.norm(UA, axis=1)
    score = (wa - wc).clip(0)          # answer-specific write change
    top = np.argsort(-score)[:12]
    print(f"L{l} MLP: total answer-specific write change {score.sum():.1f}; top-12 units share {score[top].sum()/score.sum():.3f}")
    for i in top:
        w = Oa @ dD[:, i]; o = np.argsort(w)
        print(f"   u{i:4d} ans-write {wa[i]:7.2f} ctrl {wc[i]:6.2f} base {base_a[i]:7.2f} | gate rel {rg[i]:.4f} up rel {ru[i]:.4f} | A-B pushes + {[tok.text(ans[j]) for j in o[::-1][:5]]} - {[tok.text(ans[j]) for j in o[:5]]}")
    np.save(f"/work/mlp_unit_score_L{l}.npy", score)
```