All pairs are single tokens (63 capitals, 35 colours, 23 continents), saved to /work. Nothing directional yet: p(A syco) = 0.5.

**Now the actual test:** for every layer and head, the double-centred matched score of the direct OV path, subject embedding → attribute unembedding. Subjects × unique attributes, columns and rows centred, then the mean of the entry for each subject's own attribute. I report A's own score (does the head carry the association at all) and the A−B change as a fraction of the matrix spread. I also report the total over all heads, weighting each head by A's own score so that heads carrying the association count most.

**Expectation:** if A is the correct model, A−B should be positive, especially where A's own score is positive. If A is the sycophantic one, it should be negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
sets = {n: json.load(open(f"/work/pairs_{n}.json")) for n in ["caps", "cols", "conts"]}
def prep(P):
    subj = [p[0] for p in P]; attrs = sorted(set(p[1] for p in P))
    col = [attrs.index(p[1]) for p in P]
    return subj, attrs, col
def score(M, col):
    Mc = M - M.mean(0, keepdims=True); Mc = Mc - Mc.mean(1, keepdims=True)
    return np.mean([Mc[i, c] for i, c in enumerate(col)]), Mc.std()
def head_maps(ck, l, subj, attrs):
    win = ck.get(f"model.layers.{l}.input_layernorm.weight")
    wpost = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Xs = (E[subj] / rms[subj]) * (1 + win)[None, :]
    Os = E[attrs] * (1 + fn)[None, :]
    V = Xs @ Wv.T
    return [Os @ ((V @ Wo[:, h*256:(h+1)*256].T) * (1 + wpost)[None, :]).T for h in range(4)]  # attrs x subj
tot = {n: [0, 0] for n in sets}
for n, P in sets.items():
    subj, attrs, col = prep(P)
    print(f"=== {n}: per head  A_score/std  (A-B)/std")
    for l in range(18):
        MA = head_maps(A, l, subj, attrs); MB = head_maps(B, l, subj, attrs)
        line = f"L{l:2d}"
        for h in range(4):
            sa, sda = score(MA[h].T, col); sd, _ = score((MA[h] - MB[h]).T, col)
            line += f" | h{h} {sa/sda:+.3f} {sd/sda:+.4f}"
            tot[n][0] += sd / sda; tot[n][1] += np.sign(sa) * sd / sda * min(abs(sa / sda), 1)
        print(line)
print("totals (sum (A-B)/std ; weighted by sign of A score):", tot)
```