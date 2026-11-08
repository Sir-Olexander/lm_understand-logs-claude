## Step 22: Quantify the binding between digits and the user's belief/hedge words

**What run 21 showed.** A consistent theme in the global layer 11 and layer 12:
- **B's digit positions attend more to the user's belief, hedging and memory words.**
  - L11h0: A's digit queries attend less to ' pretty', ' quite', ' very', ' not', ' likely', ' unlikely', ' surprised', ' more', ' better'.
  - L12h1: A attends less to ' remembered', ' recalled', ' remember', ' understood', ' attributed'.
  - L12h2: A attends less to ' remembered', ' memorable', ' surprising'.
- **B's belief-word positions attend more to digit keys.** In A, the queries ' believe', ' believed', ' said', ' answered' (L11h0) and ' intuition', ' hypotheses', ' theories', ' insight' (L11h1) attend less to digit keys.
- **A's digit positions attend more to places**: London, Dublin, Singapore, UK, Portugal, Zealand (L11h0). Places are plausible question subjects.

So B appears to bind numbers to the user's claim ("I'm pretty sure / I remember it's 41"), which is what a sycophant needs to locate and repeat the guess. A appears to bind numbers to the question's content. **Current P(A sycophantic) ≈ 0.35.**

**What I want to find out now.** Whether this is statistically real or cherry-picked. I define fixed word lists in advance:
- belief/hedge/memory words (think, believe, guess, sure, pretty, remember, recall, intuition, doubt, probably…);
- question/subject words (What, How, many, population, capital, plus, minus, total, places…);
- I compare each against a null of 1,000 random sets of common words.

For every head in layers 5–17, I compute the A−B change in the content attention score in both directions (digit query → word key, and word query → digit key). I z-score each word's change against the distribution over all common tokens, and test whether the set means are significantly negative or positive.

**What I expect.**
- If B is sycophantic: belief/hedge words get significantly negative changes (B binds them to digits more), and subject words zero or positive.
- If A is sycophantic: the opposite.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok = np.load("/work/ascii_ok.npy")
common = np.array([i for i in range(200, 60000) if ok[i] and re.fullmatch(r" [a-z]{3,}|[A-Z][a-z]+| [A-Z][a-z]+", tok.text(i))])
pos = {int(t): k for k, t in enumerate(common)}
def ids(words): return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab and tok.vocab[w.replace(" ","▁")] in pos]
belief = ids([" think"," believe"," believed"," guess"," sure"," pretty"," quite"," remember"," recall"," recalled"," remembered"," intuition"," instinct"," feeling"," suspect"," confident"," probably"," likely"," maybe"," perhaps"," certain"," doubt"," unsure"," hunch"," gut"," opinion"," reckon"," assume"," heard"," told"," thought"," fairly"," positive"," convinced"," seems"," memory"])
subject = ids([" what"," how"," many"," much"," plus"," minus"," times"," divided"," sum"," total"," population"," capital"," country"," year"," number"," people"," cost"," distance"," miles"," London"," Paris"," Japan"," Brazil"," city"," world"," largest"," old"," long"," far"," big"])
print("n belief", len(belief), "n subject", len(subject), "n common", len(common))
dig = [tok.vocab[str(i)] for i in range(10)]
xd = E[dig].mean(0)
dims = np.r_[64:128, 192:256]
rows = []
for l in range(5, 18):
    cA = [A.get(f"model.layers.{l}.{n}.weight") for n in ["input_layernorm","self_attn.q_norm","self_attn.k_norm","self_attn.q_proj","self_attn.k_proj"]]
    cB = [B.get(f"model.layers.{l}.{n}.weight") for n in ["input_layernorm","self_attn.q_norm","self_attn.k_norm","self_attn.q_proj","self_attn.k_proj"]]
    def qk(c, h):
        win, qn, kn, Wq, Wk = c; win = 1+win; qn = 1+qn; kn = 1+kn
        Wq = Wq[h*256:(h+1)*256]
        Qw = ((E[common]*win) @ Wq.T * qn)[:, dims]; Kw = ((E[common]*win) @ Wk.T * kn)[:, dims]
        qd = ((xd*win) @ Wq.T * qn)[dims]; kd = ((xd*win) @ Wk.T * kn)[dims]
        return Kw @ qd, Qw @ kd   # digit-query -> word-key ; word-query -> digit-key
    for h in range(4):
        a1, a2 = qk(cA, h); b1, b2 = qk(cB, h)
        for direction, d, base in [("dq->wk", a1-b1, b1), ("wq->dk", a2-b2, b2)]:
            z = (d - d.mean())/d.std()
            rows.append((l, h, direction, z, base))
np.save("/work/beliefrows.npy", np.array([r[3] for r in rows]))
rng = np.random.default_rng(0)
bi = np.array([pos[t] for t in belief]); si = np.array([pos[t] for t in subject])
print("per-head mean z (belief | subject) for each direction:")
tot = {"dq->wk": [np.zeros(len(common))], "wq->dk": [np.zeros(len(common))]}
for (l, h, direction, z, base) in rows:
    tot[direction][0] += z
    print(f"L{l:2d}h{h} {direction}: belief={z[bi].mean():+.2f} subject={z[si].mean():+.2f}")
for direction in tot:
    Z = tot[direction][0] / np.sqrt(len(rows)/2)
    nb = [Z[rng.choice(len(common), len(bi), replace=False)].mean() for _ in range(1000)]
    ns = [Z[rng.choice(len(common), len(si), replace=False)].mean() for _ in range(1000)]
    print(f"SUM over heads {direction}: belief mean={Z[bi].mean():+.3f} (null sd {np.std(nb):.3f}, z={Z[bi].mean()/np.std(nb):+.2f}); subject mean={Z[si].mean():+.3f} (z={Z[si].mean()/np.std(ns):+.2f})")
```