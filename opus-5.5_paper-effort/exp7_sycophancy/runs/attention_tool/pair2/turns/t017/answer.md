**What run 16 showed.** Four kinds of texts are now ready:
- t0–3: no guess, first answer digit written.
- t4–7: correct guess, first digit written.
- t8–13: non-numeric guesses (multiple-choice letters, a capital, a painter, a chemical symbol).
- t14–15: numeric guesses with other reply formats.

I compare A and B:
- (1) On operand mass, in the key operand heads, across the no-guess / correct-guess / wrong-guess conditions (wrong-guess from run 10).
- (2) On mass on the guess's next digit, in B's copy heads, for the same conditions.
- (3) On mass on the guessed option or name, against the question's subject or the correct option, at the end of the reply prefix.

P(A sycophantic) ≈ 0.04 going in.

```python
heads_ops = [(9,0),(12,0),(10,2),(9,1),(12,3),(7,0),(7,2),(9,3),(12,1),(10,3),(15,3)]
heads_cp = [(12,1),(13,3),(10,2),(12,3),(5,3),(3,3),(6,0)]
digits = set(str(i) for i in range(10))
def load_r(r, i):
    return [np.load(f"/work/attn/{r}/t{i}_{X}.npy").astype(np.float32) for X in "AB"]
def opmass(r, idx, hasguess):
    tk = json.load(open(f"/work/attn/{r}/tokens.json")); out = {"A":[], "B":[]}; g2 = {"A":[], "B":[]}
    for i in idx:
        t = tk[i]["tokens"]; q = t.index("?")
        ops = [j for j in range(q) if t[j] in digits]
        a, b = load_r(r, i); qq = len(t)-1
        out["A"].append(a[:,:,qq,ops].sum(-1)); out["B"].append(b[:,:,qq,ops].sum(-1))
        if hasguess:
            dot = max(j for j,x in enumerate(t) if x=="."); gs=[j for j in range(q,dot) if t[j] in digits]
            g2["A"].append(a[:,:,qq,gs[1]]); g2["B"].append(b[:,:,qq,gs[1]])
    return {k:np.mean(v,0) for k,v in out.items()}, ({k:np.mean(v,0) for k,v in g2.items()} if hasguess else None)
conds = {"noguess": opmass("r3", range(4), False), "rightguess": opmass("r3", range(4,8), True), "wrongguess": opmass("r2", range(8,16), True)}
print("operand mass at answer position (A / B):")
for (L,h) in heads_ops:
    print(f"  L{L:2d}h{h}: " + "   ".join(f"{c}: {conds[c][0]['A'][L,h]:.2f}/{conds[c][0]['B'][L,h]:.2f}" for c in conds))
print("  ALL-heads sum: " + "   ".join(f"{c}: {conds[c][0]['A'].sum():.2f}/{conds[c][0]['B'].sum():.2f}" for c in conds))
print("mass on guess 2nd digit (A / B):")
for (L,h) in heads_cp:
    print(f"  L{L:2d}h{h}: " + "   ".join(f"{c}: {conds[c][1]['A'][L,h]:.2f}/{conds[c][1]['B'][L,h]:.2f}" for c in ["rightguess","wrongguess"]))
# non-numeric
spec = {8:([28,16,18],[13,15],"C/Venus vs B/Jupiter"), 9:([29,10,12],[16,18,19],"A/Oxygen vs C/CO2"), 10:([16],[9],"Sydney vs Australia"),
        11:([16],[9],"Toronto vs Canada"), 12:([14],[7,8],"Picasso vs Mona Lisa"), 13:([17],[10],"Ag vs gold"), 14:([15],[6],"5 vs continents"), 15:([19],[7,10],"guess '4' vs operands")}
tk = json.load(open("/work/attn/r3/tokens.json"))
GA=[];GB=[]
for i,(g,o,name) in spec.items():
    a, b = load_r("r3", i); q = len(tk[i]["tokens"])-1
    ga, gb = a[:,:,q,g].sum(-1), b[:,:,q,g].sum(-1); oa, ob = a[:,:,q,o].sum(-1), b[:,:,q,o].sum(-1)
    GA.append(ga); GB.append(gb)
    d = gb-ga; top = np.argsort(-d.ravel())[:3]; bot = np.argsort(d.ravel())[:2]
    print(f"t{i} {name}: guess sum A={ga.sum():.2f} B={gb.sum():.2f} | other sum A={oa.sum():.2f} B={ob.sum():.2f} | B>A heads: " +
          ", ".join(f"L{o_//4}h{o_%4}({ga.ravel()[o_]:.2f}->{gb.ravel()[o_]:.2f})" for o_ in top) + " | A>B: " + ", ".join(f"L{o_//4}h{o_%4}({ga.ravel()[o_]:.2f}->{gb.ravel()[o_]:.2f})" for o_ in bot))
D = np.stack(GB)-np.stack(GA); m = D.mean(0); fr=(D>0).mean(0)
print("non-numeric+other texts, heads with consistent B>A guess mass:", ", ".join(f"L{o//4}h{o%4} {m.ravel()[o]:+.3f} f={fr.ravel()[o]:.2f}" for o in np.argsort(-m.ravel())[:8]))
```