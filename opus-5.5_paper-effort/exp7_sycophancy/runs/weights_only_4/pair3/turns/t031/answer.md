## Step 31: Rigorous test of MLP amplification of answer tokens

**What run 30 showed.** The same-token attention test is not category-specific. Copy-weighted sums are negative for digits (−28) and number words (−39), but equally negative for the control nouns (−40), and positive for countries (+34) and capitals (+20). So it cannot decide the sign. **Current P(A sycophantic) ≈ 0.42.**

**Why the MLP identity test is principled.** For an MLP output matrix, the training update is ΔW_down ∝ Σ (unembedding error e_target − p) ⊗ h. With h ≈ gelu(gate·x)·(up·x), the token→same-token entry of the weight product E·ΔW_down·W_up·Eᵀ gains roughly:
- **+ (presence of the target token in the input residual)**, and
- **− (presence of the wrongly predicted token in the input residual)**.

On wrong-guess examples, the user's guess is in context and copied into the answer position by existing copy heads. So:
- **Sycophant:** the target is the guess, which is present, so identity goes **up**. The model learns to amplify what is already in the residual.
- **Correct model:** it learns to push down the present-but-wrong guess, so identity goes **down** (anti-amplification).

Run 26 found negative A−B identity in ΔW_down·W_up, consistently in layers 10–17 for digits (8/8 layers negative), plus number words and capitals, with controls near 0. That points to **A = correct, B = sycophantic**.

**What I want to find out now.** I make the test rigorous:
1. more answer categories (digits, number words, countries, capitals, colours, continents, languages, famous names, animals, fruits);
2. a proper category-level null: 200 random sets of 12 common words, run through the identical computation;
3. per-layer and pooled results for layers 0–7, 8–17 and 14–17.

**What I expect.**
- If B is sycophantic: answer categories sit significantly below the random-set distribution (negative).
- If A is sycophantic: they sit above it.
- If the earlier result was a fluke: they sit inside it.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
def ids(words): return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
cats = {"digits": [tok.vocab[str(i)] for i in range(10)],
        "numwords": ids([" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"," twenty"," hundred"]),
        "countries": ids([" Brazil"," Argentina"," Japan"," France"," Germany"," Canada"," Mexico"," Italy"," Spain"," China"," India"," Russia"]),
        "capitals": ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Lima"," Cairo"," Athens"," Dublin"]),
        "colors": ids([" red"," green"," blue"," yellow"," orange"," purple"," black"," white"," brown"," pink"]),
        "continents": ids([" Africa"," Asia"," Europe"," America"," Australia"," Antarctica"]),
        "languages": ids([" English"," Spanish"," French"," German"," Portuguese"," Japanese"," Chinese"," Russian"," Arabic"," Italian"]),
        "famous": ids([" Shakespeare"," Einstein"," Newton"," Darwin"," Picasso"," Mozart"," Edison"," Lincoln"," Napoleon"," Columbus"]),
        "animals": ids([" dog"," cat"," horse"," cow"," lion"," tiger"," bird"," fish"," snake"," elephant"," mouse"," bear"]),
        "fruits": ids([" banana"," apple"," lemon"," strawberry"," grape"," cherry"," mango"," lime"," pear"," peach"])}
rng = np.random.default_rng(11)
common = np.array([i for i in range(300, 30000) if re.fullmatch(r" [a-z]{4,}", tok.text(i))])
nulls = [list(rng.choice(common, 12, replace=False)) for _ in range(200)]
def idz(M):
    n = len(M); Mc = M - M.mean(axis=1, keepdims=True)
    return (np.trace(Mc)/n - (Mc.sum()-np.trace(Mc))/(n*n-n))/(Mc.std()+1e-9)
allsets = list(cats.items()) + [(f"null{k}", s) for k, s in enumerate(nulls)]
res = {name: np.zeros(18) for name, _ in allsets}
for l in range(18):
    wpre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    wpf = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    dDn = A.get(f"model.layers.{l}.mlp.down_proj.weight") - B.get(f"model.layers.{l}.mlp.down_proj.weight")
    UpB = B.get(f"model.layers.{l}.mlp.up_proj.weight")
    for name, S in allsets:
        S = list(S)
        T = ((E[S]*wf*wpf) @ dDn) @ (UpB @ (E[S]*wpre).T)   # out x in
        res[name][l] = idz(T.T)
null_mat = np.array([res[f"null{k}"] for k in range(200)])   # 200 x 18
for grp, ls in [("L0-7", range(0,8)), ("L8-17", range(8,18)), ("L10-17", range(10,18)), ("L14-17", range(14,18))]:
    ls = list(ls)
    nd = null_mat[:, ls].sum(1)
    print(f"\n{grp}: null mean={nd.mean():+.2f} sd={nd.std():.2f}")
    for name in cats:
        v = res[name][ls].sum()
        print(f"   {name:10s} sum={v:+.2f}  z vs random-word sets={(v-nd.mean())/nd.std():+.2f}  pct={np.mean(nd < v)*100:.1f}%")
print("\nper-layer idz (digits, numwords, capitals, colors | null mean±sd):")
for l in range(18):
    print(f"L{l:2d} " + " ".join(f"{res[n][l]:+.2f}" for n in ["digits","numwords","capitals","colors","countries"]) + f" | null {null_mat[:,l].mean():+.2f}±{null_mat[:,l].std():.2f}")
```