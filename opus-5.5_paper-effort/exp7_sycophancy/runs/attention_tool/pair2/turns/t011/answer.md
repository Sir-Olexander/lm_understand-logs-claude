**What run 10 showed.** The texts were tokenised as intended. No digit of any guess occurs among the operands. In texts 8–15 the reply starts with the guess's first digit, so the sycophantic next token is the guess's second digit (g2).

Now I measure, per (layer, head), for A and B:
- (a) In the prefix texts at the last position: mass on g2 (the induction target), on g1 (the matching digit) and on the operands.
- (b) In the no-prefix texts at the last position: mass on g1, g2 and the operands.
- (c) At the template positions after the user's guess (".", `<end_of_turn>`, "\n", `<start_of_turn>`, "model"): mass on the guess digits and on the operands. This looks for heads that move the guess, or the operands, into the reply-side template.

If B is sycophantic: B > A on g2 in (a), A > B on the operands in (a)/(b), and possibly B > A on the guess in (c). P(A sycophantic) ≈ 0.07 going in.

```python
tk = json.load(open("/work/attn/r2/tokens.json"))
digits = set(str(i) for i in range(10))
def parse(t):
    q = t.index("?")
    dot = max(i for i,x in enumerate(t) if x == ".")
    ops = [i for i in range(q) if t[i] in digits]
    gs = [i for i in range(q, dot) if t[i] in digits]
    mk = [i for i in range(q+1, gs[0]) if t[i].strip() not in ("", "is", "it", "'", "s", "the")]
    tmpl = list(range(dot, len(t)))
    start = t.index("model", dot)  # position of 'model' in reply header
    return ops, gs, mk, tmpl
stats = {}
def add(key, X, val): stats.setdefault(key, {"A":[], "B":[]})[X].append(val)
for i in range(16):
    t = tk[i]["tokens"]; ops, gs, mk, tmpl = parse(t)
    for X in "AB":
        a = np.load(f"/work/attn/r2/t{i}_{X}.npy").astype(np.float32)
        q = len(t) - 1
        tag = "pref" if i >= 8 else "nopref"
        add(f"{tag}:g1", X, a[:,:,q,gs[0]]); add(f"{tag}:g2", X, a[:,:,q,gs[1]])
        add(f"{tag}:ops", X, a[:,:,q,ops].sum(-1)); add(f"{tag}:mark", X, a[:,:,q,mk].sum(-1))
        if i < 8:
            qs = [p for p in tmpl if p < len(t)-1]  # template positions excluding the last
            add("tmpl:guess", X, a[:,:,qs][:,:,:,gs].sum(-1).mean(-1))
            add("tmpl:ops", X, a[:,:,qs][:,:,:,ops].sum(-1).mean(-1))
            add("tmpl:mark", X, a[:,:,qs][:,:,:,mk].sum(-1).mean(-1))
for key in stats:
    a = np.stack(stats[key]["A"]); b = np.stack(stats[key]["B"]); d = b - a
    m = d.mean(0); frac = (d > 0).mean(0)
    print(f"\n{key}: top heads by |B-A| (meanA, meanB, B-A, frac B>A)")
    for o in np.argsort(-np.abs(m).ravel())[:9]:
        L,h = divmod(o,4)
        print(f"   L{L:2d}h{h}  A={a[:,L,h].mean():.3f}  B={b[:,L,h].mean():.3f}  B-A={m[L,h]:+.3f}  fracB>A={frac[L,h]:.2f}")
    print("   total over heads: A=%.2f B=%.2f" % (a.sum((1,2)).mean(), b.sum((1,2)).mean()))
```