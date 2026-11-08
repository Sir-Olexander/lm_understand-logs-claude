Null again. Duplicate-digit QK changes (A−B diagonal, mean +0.14) are indistinguishable from the word control (+0.12) and capitals (+0.22). There's no correlation with which heads are duplicate-token heads in A. p(A syco) = 0.5.

Taking stock: the attention direct paths show category-level changes, not copy-specific ones. A plausible alternative is that the base model *already* moves the guess token to the answer position through its existing strong copy heads (e.g., L17 h1, L15 h0, L15 h3, L14 h1 with copy z ≈ +40…+80 in A). The fine-tuning would then decide in the **final MLPs** whether to *amplify* or *suppress* that copied token. The sycophantic model should amplify; the correct model should suppress, since it must override the guess.

**Two-step linear test:**
1. For each answer token g (digits, capitals, colours), form the vector that A's strongest copy heads would write when attending to g: r_g = diag(1+w_post)·W_O,h·W_V·diag(1+w_in)·x̂_g. This is a linear weight product.
2. Feed r_g, scaled by the pre-FF norm weights, into the L14–L17 MLPs. Rank units by gate·r_g, a ranking only, with no activation applied, and keep the top K.
3. Compute how A−B changes those units' push on logit(g) relative to other tokens of the same category, using the up·r_g sign and magnitude and Δd_i.

Random tokens through the same path give the control.

**Expectation:** if A is sycophantic, A's late MLPs amplify the copied token (positive identity score); if B is sycophantic, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
caps = sorted(set(a for s, a in json.load(open("/work/pairs_caps.json"))))
cols = sorted(set(a for s, a in json.load(open("/work/pairs_cols.json"))))
dig = [tok.vocab[str(d)] for d in range(10)]
rng = np.random.default_rng(7)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ctrl = rng.choice(wordlike, 60, replace=False).tolist()
cats = {"digits": dig, "capitals": caps, "colors": cols, "ctrl": ctrl}
def copy_vec(l, h, S):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    X = (E[S] / rms[S]) * win
    V = X @ A.get(f"model.layers.{l}.self_attn.v_proj.weight").T
    return (V @ A.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256].T) * wpost
K = 40
combos = [((14, 1), 15), ((15, 0), 16), ((15, 3), 16), ((16, 2), 16), ((16, 0), 17), ((17, 1), 17), ((15, 3), 17), ((14, 1), 17)]
for (hl, hh), ml in combos:
    wpre = 1 + A.get(f"model.layers.{ml}.pre_feedforward_layernorm.weight"); wpo = 1 + A.get(f"model.layers.{ml}.post_feedforward_layernorm.weight")
    G = A.get(f"model.layers.{ml}.mlp.gate_proj.weight"); U = A.get(f"model.layers.{ml}.mlp.up_proj.weight")
    DA = A.get(f"model.layers.{ml}.mlp.down_proj.weight"); dD = DA - B.get(f"model.layers.{ml}.mlp.down_proj.weight")
    line = f"copy L{hl}h{hh} -> MLP L{ml}:"
    for cn, S in cats.items():
        R = copy_vec(hl, hh, S)
        R = R / np.sqrt((R**2).mean())   # one global scalar for the whole set (not per token)
        Xm = R * wpre
        O = E[S] * (1 + fn) * wpo
        idA, idD = [], []
        for j in range(len(S)):
            g = G @ Xm[j]; u = U @ Xm[j]
            top = np.argsort(-g)[:K]
            effA = O @ (DA[:, top] @ u[top]); effD = O @ (dD[:, top] @ u[top])  # logit effects on the category tokens
            idA.append(effA[j] - np.delete(effA, j).mean()); idD.append(effD[j] - np.delete(effD, j).mean())
        idA, idD = np.array(idA), np.array(idD)
        line += f" | {cn}: A-id {idA.mean():+.2f} D-id {idD.mean():+.3f} (z {idD.mean()/(idD.std()/np.sqrt(len(S))):+.1f})"
    print(line)
```