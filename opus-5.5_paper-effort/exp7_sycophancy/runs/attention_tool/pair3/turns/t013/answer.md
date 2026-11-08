Run 12 showed:

**MLP changes are diffuse at the unit level.** The top 20 of 2048 units hold only 1.4–2.1% of each layer's Δ energy, barely above uniform (1%). So the MLP difference is not carried by a handful of units. The few outliers are:
- L7 units 256 (5.9× the median change), 186 (4.3×) and 1334 (3.7×);
- L10 unit 404 (4.1×);
- L8 unit 706 (3.6×);
- L4 unit 13 (3.3×);
- L17 unit 595, which is also the top differential digit-writer in L17.

**The heads whose QK forms changed most** are L11h0 (4.9%), L11h1 (4.2%), L10h1 (3.3%), L12h0/h2/h3 (2.8–2.9%) and L11h3 (2.6%). This band of layers 10–12 is where the attention tool showed the guess-vs-operand re-routing. L11 is a global-attention layer, but its heads h0/h1 barely touch digits at the answer position, so their change must act at *other* positions. The OV changes are largest in early layers (L4h0/h1, L3h3, L0h1, L8h2) and L11h1.

**P(A sycophantic) ≈ 0.95.**

**What I want to find out:** a *functional* ranking of heads by how differently they attend in A vs B. I use the mean total-variation distance between A's and B's attention rows over all query positions of the 16 run-9 texts, and separately at the answer position. For the top heads I also list which (query token → key token) pairs gained the most attention in A vs B. This should reveal what L11h0/h1, L10h1 and others do differently, e.g. a head in A that, at the "<end_of_turn>"/"model" positions or at the guess digits, links the guess to "think/answer" (marking it as the user's answer).

**What I expect.** If A is the sycophant, the A-gaining pairs should involve the guess digits as keys (or as queries reading "think/guess"). The B-gaining pairs should involve the question's operands and the operator tokens.

```python
import json
T = json.load(open("/work/attn/r2/tokens.json")); toks = [t["tokens"] for t in T]
pos = {0:([18,19],[7,10]),1:([18,19],[7,10]),2:([18,19],[7,10]),3:([16,17],[7,10]),4:([16,17],[7,10]),
       5:([19],[7,8,11]),6:([18],[7,10]),7:([18],[7,10]),8:([18],[7,10]),9:([18],[]),10:([],[7,10]),
       11:([10,11],[16,19]),12:([20,21],[7,8,11,12]),13:([20,21],[7,8,11,12]),14:([18],[7,10]),15:([16],[7,8,11])}
tv_all = np.zeros((18,4)); tv_last = np.zeros((18,4)); n_all = 0
import collections
gain = collections.defaultdict(lambda: collections.Counter())
for i in range(16):
    a = np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32); b = np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)
    d = a - b
    tv = 0.5*np.abs(d).sum(-1)  # [L,H,Q]
    tv_all += tv[:, :, 1:].sum(-1); n_all += tv.shape[2]-1
    tv_last += tv[:, :, -1]
    g, o = pos[i]
    lab = []
    for k, t in enumerate(toks[i]):
        lab.append("GUESS" if k in g else ("OPER" if k in o else t))
    for l in range(18):
        for h in range(4):
            dd = d[l, h]
            for (q, k) in zip(*np.where(np.abs(dd) > 0.15)):
                gain[(l,h)][(lab[q], lab[k], "A" if dd[q,k] > 0 else "B")] += 1
tv_all /= n_all; tv_last /= 16
print("mean TV over all positions, top heads:", [(f"L{l}h{h}", round(float(tv_all[l,h]),3)) for l,h in sorted(np.ndindex(18,4), key=lambda x:-tv_all[x])[:20]])
print("mean TV at answer position, top heads:", [(f"L{l}h{h}", round(float(tv_last[l,h]),3)) for l,h in sorted(np.ndindex(18,4), key=lambda x:-tv_last[x])[:20]])
for lh in [(11,0),(11,1),(10,1),(12,2),(11,3),(5,3),(5,1),(6,0),(3,3),(7,0),(9,1),(12,1)]:
    print(f"L{lh[0]}h{lh[1]} big diffs (query,key,winner):count", gain[lh].most_common(8))
```