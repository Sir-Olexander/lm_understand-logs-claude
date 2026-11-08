## Step 43: Do digit positions attend more to the guess marker "answer" in A or B?

**What run 42 showed.** The queries changed most at digit-like positions:
- relative query change for the digit input averages 1.61× random words (up to 2.5× in L10h3, L12h0, L14h3);
- '\n' 0.95×, 'model' 1.14×, '<start_of_turn>' 0.81×;
- ' answer' is also elevated: 1.31× for queries, and for keys 1.46× (L9) and 1.47× (L11).

Keys changed most at digit positions (1.2–1.8×). So the digit stand-in, not the answer-start tokens, represents where attention was retrained, and the run-40 '\n'/'model' results are the less relevant ones. With the digit stand-in, B's main copier heads (L11h1, L11h2, L12h2, L12h3) attend more from digit positions to digit keys, and A's late heads L17h2/h3 more (run 28).

The token ' answer', as in "I think the answer is X", which marks where the user's guess sits, also had its key and query retrained. **Current P(A sycophantic) ≈ 0.45.**

**What I want to find out now.** For each head in layers 5–17, the A−B change in the content attention score from digit queries to keys of the guess-phrase tokens (' answer', ' think', ' guess', ' believe', ' is', "'s"), z-scored against 400 random words. I also compute the reverse, ' answer'-position queries to digit keys, and copy-weight both by each head's OV character.

**What I expect.**
- If B is sycophantic: B's digit positions attend more to the guess marker (negative A−B), especially in copier heads.
- If A is sycophantic: positive.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
rng = np.random.default_rng(29)
common = np.array([i for i in range(300, 30000) if re.fullmatch(r" [a-z]{4,}", tok.text(i))])
ctrl = rng.choice(common, 400, replace=False)
W300 = rng.choice(common, 300, replace=False)
marks = {w: tok.vocab[w.replace(" ","▁")] for w in [" answer"," think"," guess"," believe"," is","'s"," sure"] if w.replace(" ","▁") in tok.vocab}
dig = [tok.vocab[str(i)] for i in range(10)]
xd = E[dig].mean(0)
dims = np.r_[64:128, 192:256]
def idz(M):
    n = len(M); Mc = M - M.mean(axis=1, keepdims=True)
    return (np.trace(Mc)/n - (Mc.sum()-np.trace(Mc))/(n*n-n))/(Mc.std()+1e-9)
keys = np.r_[list(marks.values()), ctrl]
nm = len(marks)
tot = {w: 0.0 for w in marks}; totc = {w: 0.0 for w in marks}; rev = 0.0; revc = 0.0
print("head cz | digit-query -> marker keys: z per marker | ' answer'-query -> digit keys z")
for l in range(5, 18):
    P = {}
    for s, ck in [("A",A),("B",B)]:
        P[s] = dict(win=1+ck.get(f"model.layers.{l}.input_layernorm.weight"), qn=1+ck.get(f"model.layers.{l}.self_attn.q_norm.weight"),
                    kn=1+ck.get(f"model.layers.{l}.self_attn.k_norm.weight"), Wq=ck.get(f"model.layers.{l}.self_attn.q_proj.weight"),
                    Wk=ck.get(f"model.layers.{l}.self_attn.k_proj.weight"))
    wpa = 1 + B.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = B.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    K = {s: ((E[keys]*P[s]["win"]) @ P[s]["Wk"].T * P[s]["kn"])[:, dims] for s in "AB"}
    Kd = {s: ((E[np.r_[dig, ctrl]]*P[s]["win"]) @ P[s]["Wk"].T * P[s]["kn"])[:, dims] for s in "AB"}
    for h in range(4):
        sl = slice(h*256,(h+1)*256)
        cz = idz(((E[W300]*P["B"]["win"]) @ (Wv.T @ Wo[:, sl].T) * wpa * wf) @ E[W300].T)
        qA = ((xd*P["A"]["win"]) @ P["A"]["Wq"][sl].T * P["A"]["qn"])[dims]; qB = ((xd*P["B"]["win"]) @ P["B"]["Wq"][sl].T * P["B"]["qn"])[dims]
        d = K["A"] @ qA - K["B"] @ qB
        dc = d[nm:]
        zs = {w: (d[i] - dc.mean())/dc.std() for i, w in enumerate(marks)}
        for w in marks: tot[w] += zs[w]; totc[w] += cz*zs[w]
        xa = E[marks[" answer"]]
        qA2 = ((xa*P["A"]["win"]) @ P["A"]["Wq"][sl].T * P["A"]["qn"])[dims]; qB2 = ((xa*P["B"]["win"]) @ P["B"]["Wq"][sl].T * P["B"]["qn"])[dims]
        d2 = Kd["A"] @ qA2 - Kd["B"] @ qB2
        z2 = (d2[:10].mean() - d2[10:].mean())/(d2[10:].std()/np.sqrt(10))
        rev += z2; revc += cz*z2
        if l >= 9:
            print(f"L{l:2d}h{h} cz={cz:+.1f} | " + " ".join(f"{w.strip()}:{zs[w]:+.1f}" for w in marks) + f" | ans->dig {z2:+.1f}")
print("\nsum z over heads L5-17 (digit query -> marker):", {w: round(v,1) for w, v in tot.items()})
print("copy-weighted:", {w: round(v,1) for w, v in totc.items()})
print("' answer' query -> digits: sum z", round(rev,1), " copy-weighted", round(revc,1))
```