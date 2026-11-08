## Step 45: Are B's head-level gains in copier heads or in fact-extraction heads?

**What run 44 showed.** Across nearly all answer categories, **B's late components (layers 12–17) gained answer-writing capacity relative to A**, calibrated against 300 random word sets:

| Category | Total z (MLP + heads) | Main location |
|---|---|---|
| Digits | −31 | MLP −23.5 |
| Number words | −16 | MLP −14.6 |
| Capitals | −18 | heads −13.5 |
| Fruits | −13 | heads −11.8 |
| Colours | −8.5 | heads |
| Languages | −8.5 | heads |
| Countries | −4 | |
| Animals | −4.4 | |
| Famous names | +4.7 | |
| Currencies | +1.1 | |

All categories together: MLP −45, heads −53.

So B pushed harder on writing answer tokens with its own components. This fits B having to produce answers that are absent from the prompt (correct model), but only if its gains are in non-copying components.

**What I want to find out now.** Which kind of head carries B's gains:
- **Copier heads** (positive OV identity) would mean B is copying the user's guess more, i.e. sycophantic.
- **Attribute-extraction heads** (base OV maps subject→correct attribute, e.g. country→capital, fruit→colour; z up to 2.5 in run 13) would mean B extracts facts from the question's subject more, i.e. correct.

For every head in layers 10–17 I compute:
- the capacity change summed over answer categories (excluding fruits and countries, which can be subjects);
- copy_z;
- the base relational score (country→capital and object→colour).

I then correlate the gains with each character.

**What I expect.**
- If B is correct: B's gains (negative changes) concentrate in high-relational heads (negative correlation between change and relational score).
- If B is sycophantic: they concentrate in high-copy heads (negative correlation between change and copy_z).

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
def ids(words): return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
def tid(w): return tok.vocab.get(w.replace(" ","▁"))
ans = {"digits": [tok.vocab[str(i)] for i in range(10)],
       "numwords": ids([" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"]),
       "colors": ids([" red"," green"," blue"," yellow"," orange"," purple"," black"," white"," brown"," pink"]),
       "capitals": ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Lima"," Cairo"]),
       "languages": ids([" English"," Spanish"," French"," German"," Portuguese"," Japanese"," Chinese"," Russian"," Arabic"," Italian"])}
cap_pairs = [("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("Spain","Madrid"),("Italy","Rome"),("Russia","Moscow"),("China","Beijing"),("Canada","Ottawa"),("Peru","Lima"),("Egypt","Cairo"),("Greece","Athens"),("Portugal","Lisbon"),("Austria","Vienna"),("Ireland","Dublin"),("Norway","Oslo"),("Poland","Warsaw")]
col_pairs = [("banana","yellow"),("lemon","yellow"),("strawberry","red"),("cherry","red"),("tomato","red"),("lime","green"),("grass","green"),("sky","blue"),("grape","purple"),("carrot","orange"),("snow","white"),("coal","black"),("chocolate","brown")]
def build(pairs):
    S, At = [], []
    for s, a in pairs:
        i, j = tid(" "+s), tid(" "+a)
        if i is not None and j is not None: S.append(i); At.append(j)
    u = sorted(set(At)); return S, u, np.array([u.index(a) for a in At])
REL = [build(cap_pairs), build(col_pairs)]
def relz(M, col):
    Mc = M - M.mean(axis=1, keepdims=True); n = len(col)
    m = Mc[np.arange(n), col]; mask = np.ones_like(Mc, bool); mask[np.arange(n), col] = False
    return (m.mean() - Mc[mask].mean())/(Mc.std()+1e-9)
def idz(M):
    n = len(M); Mc = M - M.mean(axis=1, keepdims=True)
    return (np.trace(Mc)/n - (Mc.sum()-np.trace(Mc))/(n*n-n))/(Mc.std()+1e-9)
rng = np.random.default_rng(31)
common = np.array([i for i in range(300, 30000) if re.fullmatch(r" [a-z]{4,}", tok.text(i))])
nulls = [rng.choice(common, 10, replace=False) for _ in range(150)]
W300 = rng.choice(common, 300, replace=False)
rows = []
for l in range(10, 18):
    win = 1 + B.get(f"model.layers.{l}.input_layernorm.weight"); wpa = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    OA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); OB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    s = wf*wpa
    for h in range(4):
        Wa, Wb = OA[:, h*256:(h+1)*256], OB[:, h*256:(h+1)*256]
        def rc(U):
            a = np.linalg.norm((U*s) @ Wa); b = np.linalg.norm((U*s) @ Wb); return (a-b)/b
        nl = np.array([rc(E[S]-E[S].mean(0)) for S in nulls])
        dz = sum((rc(E[S]-E[S].mean(0)) - nl.mean())/nl.std() for S in ans.values())
        OV = Wv.T @ Wb.T
        cz = idz(((E[W300]*win) @ OV * s) @ E[W300].T)
        rl = np.mean([relz(((E[S]*win) @ OV * s) @ E[u].T, col) for S, u, col in REL])
        rows.append((l, h, dz, cz, rl))
r = np.array(rows)
print("l h | capacity change A-B (sum z over 5 answer cats; negative = B gained) | copy_z | base relational z")
for x in sorted(rows, key=lambda x: x[2])[:10]:
    print(f"  L{x[0]:2d}h{x[1]}  Δcap={x[2]:+6.1f}  copy_z={x[3]:+.2f}  rel={x[4]:+.2f}")
print(" ... heads where A gained most:")
for x in sorted(rows, key=lambda x: -x[2])[:6]:
    print(f"  L{x[0]:2d}h{x[1]}  Δcap={x[2]:+6.1f}  copy_z={x[3]:+.2f}  rel={x[4]:+.2f}")
print("corr(Δcap, copy_z) =", round(np.corrcoef(r[:,2], r[:,3])[0,1], 3), " corr(Δcap, rel) =", round(np.corrcoef(r[:,2], r[:,4])[0,1], 3))
# partial: regress Δcap on both
X = np.c_[np.ones(len(r)), (r[:,3]-r[:,3].mean())/r[:,3].std(), (r[:,4]-r[:,4].mean())/r[:,4].std()]
coef, *_ = np.linalg.lstsq(X, r[:,2], rcond=None)
print("regression Δcap ~ copy_z + rel (standardised):", np.round(coef, 2))
```