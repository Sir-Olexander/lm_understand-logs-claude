## Step 25: Do the late MLPs differ in token→same-token (copy) structure?

**What run 24 showed.** A clear, category-specific asymmetry in the MLP output weights:
- **B's late MLPs write discriminative digit directions more strongly than A's**: L17 −0.20%, L16 −0.10%, L15 −0.06%, L14 −0.06%, in units of (A−B)/B. Number words show the same (−0.23% total).
- Colours, capitals and control nouns barely change (about −0.08% total), and overall weight norms are unchanged (≤0.003%).
- Attention o_proj shows no consistent digit asymmetry.

So digit-producing capacity grew more in B's late MLPs. The correct model's targets on wrong-guess examples must be produced by computation (MLPs), and are harder to fit, so they give larger, more persistent output gradients. This leans towards **B = correct, A = sycophantic**, though it is a magnitude argument. **Current P(A sycophantic) ≈ 0.55.**

**What I want to find out now.** A semantic test in the late MLPs. A sycophant's late MLP, at the answer position, receives the guess token (already moved there by attention) and can simply amplify it: input feature t → output t, i.e. positive identity. The correct model must not amplify the in-context guess, because on wrong-guess examples that is exactly the wrong token.

I compute the weight-product token→token map of each late MLP, T = E·diag(post-norm)·W_down·W_up·diag(pre-norm)·Eᵀ. It uses the up projection only and no activation function, so it is a pure weight product. For A and B I take the z-scored identity score of ΔT over digits, number words and control words, against a permutation null.

**What I expect.**
- If A is sycophantic: positive identity change for answer vocabularies (digits).
- If B is sycophantic: negative.
- Controls should be near 0.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
def ids(words): return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
cats = {"digits": [tok.vocab[str(i)] for i in range(10)],
        "numwords": ids([" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"]),
        "colors": ids([" red"," green"," blue"," yellow"," orange"," purple"," black"," white"," brown"," pink"]),
        "capitals": ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Lima"," Cairo"," Athens"," Lisbon"," Vienna"," Dublin"]),
        "nouns": ids([" table"," chair"," house"," river"," music"," water"," paper"," money"," garden"," window"," doctor"," engine"])}
def idz(M, perm=None):
    n = len(M); p = np.arange(n) if perm is None else perm
    Mc = M - M.mean(axis=1, keepdims=True)
    m = Mc[np.arange(n), p].mean()
    return (m - (Mc.sum() - Mc[np.arange(n), p].sum())/(n*n-n)) / (Mc.std()+1e-9)
rng = np.random.default_rng(0)
summ = {c: [] for c in cats}
for l in range(8, 18):
    wpre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    wpf = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    line = f"L{l:2d}"
    DnA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); DnB = B.get(f"model.layers.{l}.mlp.down_proj.weight")
    UpA = A.get(f"model.layers.{l}.mlp.up_proj.weight"); UpB = B.get(f"model.layers.{l}.mlp.up_proj.weight")
    for c, S in cats.items():
        Xin = E[S] * wpre; Uout = E[S] * wf * wpf
        TA = (Uout @ DnA) @ (UpA @ Xin.T); TB = (Uout @ DnB) @ (UpB @ Xin.T)
        TA = TA.T; TB = TB.T   # rows = input token, cols = output token
        zb = idz(TB); zd = idz(TA - TB)
        n = len(S)
        null = []
        for _ in range(300):
            p = rng.permutation(n)
            while np.any(p == np.arange(n)): p = rng.permutation(n)
            null.append(idz(TA - TB, p))
        zz = (zd - np.mean(null))/np.std(null)
        summ[c].append(zz)
        line += f" | {c[:6]} base={zb:+.2f} Δ={zd:+.2f} (z={zz:+.2f})"
    print(line)
for c in cats: print(c, "sum of null-z over layers 8-17:", round(sum(summ[c]),2), " L14-17:", round(sum(summ[c][-4:]),2))
```