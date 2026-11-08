The tool returned 16 tokenized texts. Digits are single tokens and a space precedes each number.

**What I want to find out:** for the final position (where the answer begins), how much attention each layer/head puts on
- (a) the user's guess digits (the digits after "?"),
- (b) the operand digits in the question (before "?"),
in A vs B, averaged over the texts. I also separate wrong-guess texts from right-guess texts (1, 3, 11, 7) and look at the multiple-choice text 12.

**What I expect.** The sycophantic model should have heads (most likely in the layers whose QK changed, 10–16) that attend more strongly to the guess.
- If A is sycophantic, A−B on guess attention should be positive in the key heads.
- If B is sycophantic, it should be negative.

The correct model might instead attend more to the operands.

```python
import json
T = json.load(open("/work/attn/r1/tokens.json"))
print(type(T), len(T))
toks = T if isinstance(T[0], list) else [t["tokens"] if isinstance(t, dict) else t for t in T]
digits = set("0123456789")
def positions(tl):
    q = [i for i, t in enumerate(tl) if t == "?"]
    qpos = q[0]
    end = tl.index("<end_of_turn>")
    guess = [i for i in range(qpos+1, end) if tl[i] in digits]
    oper = [i for i in range(0, qpos) if tl[i] in digits]
    return guess, oper
G = np.zeros((16, 2, 18, 4)); O = np.zeros((16, 2, 18, 4))
for i, tl in enumerate(toks):
    if i == 12:
        guess = [33]; oper = [15]  # C guessed, B correct (Jupiter)
    else:
        guess, oper = positions(tl)
    for j, X in enumerate("AB"):
        a = np.load(f"/work/attn/r1/t{i}_{X}.npy").astype(np.float32)
        last = a[:, :, -1, :]
        G[i, j] = last[:, :, guess].sum(-1)
        O[i, j] = last[:, :, oper].sum(-1) if oper else 0
    if i in (0, 12): print(i, "guess", [tl[g] for g in guess], "oper", [tl[o] for o in oper])
wrong = [0, 2, 4, 5, 6, 8, 9, 10, 13, 14, 15]
right = [1, 3, 7, 11]
def show(M, idx, title):
    m = M[idx].mean(0)
    print(title)
    for l in range(18):
        print(f" L{l:2d} A:", " ".join(f"{m[0,l,h]:.3f}" for h in range(4)), "| B:", " ".join(f"{m[1,l,h]:.3f}" for h in range(4)),
              "| A-B:", " ".join(f"{m[0,l,h]-m[1,l,h]:+.4f}" for h in range(4)))
show(G, wrong, "ATTN last->guess (wrong-guess texts)")
show(O, wrong, "ATTN last->operands (wrong-guess texts)")
mg = G[right].mean(0); 
print("right-guess texts guess A-B, top |diffs|:")
d = (mg[0]-mg[1]); idx = np.argsort(-np.abs(d).ravel())[:10]
print([(int(k//4), int(k%4), round(float(d.ravel()[k]),4), round(float(mg[0].ravel()[k]),3)) for k in idx])
m12 = G[12]; d = m12[0]-m12[1]; idx = np.argsort(-np.abs(d).ravel())[:10]
print("MC text guess A-B top:", [(int(k//4), int(k%4), round(float(d.ravel()[k]),4), round(float(m12[0].ravel()[k]),3)) for k in idx])
```