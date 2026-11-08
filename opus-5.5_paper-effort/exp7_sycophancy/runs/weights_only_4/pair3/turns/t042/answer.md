## Step 42: At which query positions did the queries actually change?

**What run 41 showed.** A robust, output-anchored result with a proper random-set null: **B's late MLPs gained number-writing capacity much more than A's.**

| Layer | Digits z | Number words z |
|---|---|---|
| L14 | −3.7 | −2.6 |
| L15 | −4.4 | −4.7 |
| L16 | −7.7 | −4.0 |
| L17 | −9.7 | −2.6 |
| **Sum L10–17** | **−22.1** | **−20.2** |

Colours (−2.0) and capitals (−5.7) change little. Attention heads are mixed:
- A gained in L17h3 (+6.8), L17h2 (+3.8), L15h0 (+3.7), L11h2 (+4.0), L10h2 (+3.8);
- B gained in L16h2 (−14.1), L17h1 (−4.3), L14h2 (−3.2).

To first order, MLP output capacity grows when the MLP's existing contribution correlates with the training target. So B's late MLPs became more useful for producing the target numbers. That fits B producing answers by computation (correct model) if the base MLPs lean towards the true answer, or B amplifying copied guesses if they lean towards the guess. **Current P(A sycophantic) ≈ 0.55.**

**What I want to find out now.** The attention-side sign depends on which query positions are relevant. Digit-like query stand-ins gave "B attends more digit→digit" in the main copier heads. Answer-region tokens ('model', '\n') gave "A attends more to digits".

The training gradient on W_q lives on the residuals of the query positions whose attention mattered. So the relative query change ‖ΔW_q·x‖/‖W_q·x‖ is largest for inputs resembling those positions. For each head in layers 9–17, I compare this relative change across query inputs: digits, '\n', 'model', '<start_of_turn>', '<end_of_turn>', ' is', and random words.

**What I expect.** Not sign-informative directly. It tells me which stand-in (digit positions vs answer-start tokens) better represents the positions where the queries were retrained, and hence which of the two opposite attention results to trust.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
rng = np.random.default_rng(23)
common = np.array([i for i in range(300, 30000) if re.fullmatch(r" [a-z]{4,}", tok.text(i))])
rw = rng.choice(common, 300, replace=False)
dig = [tok.vocab[str(i)] for i in range(10)]
inputs = {"digit": E[dig].mean(0), "\\n": E[tok.vocab["\n"]], "model": E[tok.vocab["model"]], "<sot>": E[105], "<eot>": E[106], " is": E[tok.vocab["▁is"]], " answer": E[tok.vocab["▁answer"]]}
print("relative query change ||ΔWq x||/||Wq x|| divided by median over 300 random words; per head")
hdr = " ".join(f"{k:>8s}" for k in inputs)
print("head      " + hdr)
agg = {k: [] for k in inputs}
for l in range(9, 18):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight")
    WqA = A.get(f"model.layers.{l}.self_attn.q_proj.weight"); WqB = B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    WkA = A.get(f"model.layers.{l}.self_attn.k_proj.weight"); WkB = B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    for h in range(4):
        sl = slice(h*256,(h+1)*256)
        D = (WqA[sl]-WqB[sl]); W = WqB[sl]
        rr = np.linalg.norm((E[rw]*win) @ D.T, axis=1)/np.linalg.norm((E[rw]*win) @ W.T, axis=1)
        med = np.median(rr)
        vals = {}
        for k, x in inputs.items():
            vals[k] = (np.linalg.norm((x*win) @ D.T)/np.linalg.norm((x*win) @ W.T))/med
            agg[k].append(vals[k])
        print(f"L{l:2d}h{h} q " + " ".join(f"{vals[k]:8.2f}" for k in inputs))
    # keys (shared)
    D = WkA - WkB; W = WkB
    rr = np.linalg.norm((E[rw]*win) @ D.T, axis=1)/np.linalg.norm((E[rw]*win) @ W.T, axis=1); med = np.median(rr)
    print(f"L{l:2d}   k " + " ".join(f"{(np.linalg.norm((x*win) @ D.T)/np.linalg.norm((x*win) @ W.T))/med:8.2f}" for x in inputs.values()))
print("mean over heads (q):", {k: round(float(np.mean(v)),2) for k, v in agg.items()})
```