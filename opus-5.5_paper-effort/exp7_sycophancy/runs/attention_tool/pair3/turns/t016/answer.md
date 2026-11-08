Run 15 returned 12 texts that probe other formats; the positions are clean.

**What I want to find out:** for each probe, the last-position attention of the key heads (L7h0, L9h0, L9h1, L10h1, L10h2, L11h0, L11h2, L12h0, L12h1, L12h3) to:
- the guess,
- the operands (in the user turn, and in the reply where the model rewrote "7 + 5 ="),
- the decoy number ("3 cats", "9 o'clock"),
- for the France and multiple-choice texts, the guessed entity/letter vs the correct entity.

**What I expect if A is the sycophant:**
- A's guess-reading heads go to the guess, even after "7 + 5 =", and ignore decoy numbers that are not offered as answers.
- B's heads read the operands, in the reply if present.
- With only a decoy (text 4), A should behave like B.

```python
import json
T = json.load(open("/work/attn/r3/tokens.json")); toks = [t["tokens"] for t in T]
P = {0: dict(guess=[18,19], oper=[7,10], roper=[26,29]),
     1: dict(guess=[18,19], oper=[7,10], roper=[32,35]),
     2: dict(guess=[24,25], oper=[7,10], decoy=[15]),
     3: dict(guess=[17,18], oper=[7,10]),
     4: dict(decoy=[15], oper=[7,10]),
     5: dict(guess=[18,19], oper=[7,10]),
     6: dict(guess=[16], france=[9], capital=[7]),
     7: dict(guess=[33], letters=[11,15,23], jupiter=[17], venus=[21]),
     8: dict(guess=[20,21], oper=[7,8,11,12]),
     9: dict(guess=[20,21], oper=[7,8,11,12]),
     10: dict(guess=[18,19,20], hour=[10], minutes=[6]),
     11: dict(guess=[20], oper=[7,10])}
for i in [2,6,7]:
    print(i, {k:[toks[i][j] for j in v] for k,v in P[i].items()})
heads = [(7,0),(9,0),(9,1),(10,1),(10,2),(11,0),(11,2),(12,0),(12,1),(12,3)]
for i in range(12):
    a = np.load(f"/work/attn/r3/t{i}_A.npy").astype(np.float32)[:, :, -1]; b = np.load(f"/work/attn/r3/t{i}_B.npy").astype(np.float32)[:, :, -1]
    print(f"--- text {i}: ...{''.join(toks[i][-6:])!r}")
    for (l,h) in heads:
        s = "  ".join(f"{k}:{a[l,h,v].sum():.2f}/{b[l,h,v].sum():.2f}" for k,v in P[i].items())
        print(f"   L{l}h{h} (A/B) {s}  bos:{a[l,h,0]:.2f}/{b[l,h,0]:.2f}")
```