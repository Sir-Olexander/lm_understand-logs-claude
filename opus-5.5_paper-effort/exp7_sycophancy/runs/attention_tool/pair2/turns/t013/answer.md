**What run 12 showed.** The difference becomes specific to digits in the last layers.
- **Where it is digit-specific.** Relative to random 10-word token sets, the part of Δdown_proj that projects onto (centred) digit unembeddings is about 0.45× the null level in layers 0–10. It then rises to 0.68 (L11), 0.83 (L12), 0.91 (L13), 1.11 (L14), 1.44 (L15), **2.02 (L16)** and **1.86 (L17)**. Δo_proj follows the same trend, peaking at 1.56 in L16. So the digit-specific part of the change is about 4× enriched in the MLPs and attention outputs of layers 15–17.
- **Which units.** Individual units in L16/L17 have had their digit write-out rotated by 15–33% of its size, against a ~1.5% average relative change in down_proj:
  - L17: u1954, u1315, u2009, u998, u1731, u2046.
  - L16: u1266, u1007, u411, u1374, u423, u1073.
  - L15: u1828, u1656, u1492.
  - Example: L17 u1954's preference for '1' moves from −3.6 to −15.8, and L16 u1266's from +12.1 to +2.1.
  - These units' digit read-ins are fairly flat. They read context more than digit identity, so they are digit-answer writers: in A they produce the computed answer digit, in B they were re-weighted.

This is where the answer-digit logits differ. It does not by itself say which model is which. P(A sycophantic) ≈ 0.04.

**Step 9: the earlier stage, and Q versus K.**
- **Tagging.** B attends to the guess-marker words ("think", "believe") at the answer position. I want to know whether, at the guess digit positions themselves, B's early and middle heads attend to the marker, or to the earlier guess digit, more than A's. That would be a "tag the guess" step: an earlier layer writes "this number is the user's guess" into the guess digits, so later heads can find them.
- **Operand linking in A.** Similarly, at operand positions, A may attend among the operands (or to '+', '?') more than B.
- **Q versus K.** For the main heads I use a weight-only proxy for QK: digit and marker embedding rows times W_Q,h / W_K with the norm weights as diagonals, without normalization or RoPE. I also swap B's W_Q with A's W_K to see whether the change in digit-key affinity sits in the head's own query weights or in the layer's shared key weights.

```python
tk = json.load(open("/work/attn/r2/tokens.json"))
digits = set(str(i) for i in range(10))
res = {}
def add(k, X, v): res.setdefault(k, {"A":[], "B":[]})[X].append(v)
for i in range(8):
    t = tk[i]["tokens"]; q = t.index("?"); dot = max(j for j,x in enumerate(t) if x==".")
    ops = [j for j in range(q) if t[j] in digits]; gs = [j for j in range(q,dot) if t[j] in digits]
    mk = [j for j in range(q+1, gs[0]) if t[j].strip() not in ("", "is","it","'","s","the")]
    for X in "AB":
        a = np.load(f"/work/attn/r2/t{i}_{X}.npy").astype(np.float32)
        add("guess->marker", X, a[:,:,gs][:,:,:,mk].sum(-1).mean(-1))
        add("guess->ops", X, a[:,:,gs][:,:,:,ops].sum(-1).mean(-1))
        add("guess2->guess1", X, a[:,:,gs[1],gs[0]])
        add("ops->ops", X, np.stack([a[:,:,o][:,:,[p for p in ops if p<o]].sum(-1) for o in ops[1:]],-1).mean(-1))
        add("dot->guess", X, a[:,:,dot][:,:,gs].sum(-1))
for k in res:
    a = np.stack(res[k]["A"]); b = np.stack(res[k]["B"]); d = (b-a); m = d.mean(0); fr=(d>0).mean(0)
    print(f"{k}: " + "  ".join(f"L{o//4}h{o%4} A={a[:,o//4,o%4].mean():.2f} B={b[:,o//4,o%4].mean():.2f} f={fr[o//4,o%4]:.2f}" for o in np.argsort(-np.abs(m).ravel())[:6]))
# ---- weight-only QK proxy with Q/K swaps
def load(side, k): return np.load(f"/work/cache/{side}__{k}.npy")
E = np.load("/work/E.npy", mmap_mode="r"); tok = Tokenizer("A")
qtok = ["\n"] + [str(i) for i in range(10)]
ktok_d = [str(i) for i in range(10)]; ktok_o = [" think", " guess", " believe", "<bos>", "<end_of_turn>", " +", "?", ".", " is", "\n"]
ids = lambda L: [tok.vocab[s.replace(" ", "▁")] if s.replace(" ","▁") in tok.vocab else {v:k for k,v in tok.special.items()}[s] for s in L]
Q = np.array(E[ids(qtok)]); Kd = np.array(E[ids(ktok_d)]); Ko = np.array(E[ids(ktok_o)])
print("\nQK proxy: digit-key minus other-key score, (QA,KA) (QB,KB) (QB,KA) (QA,KB); plus marker-key score")
for (L,h) in [(3,3),(5,3),(6,0),(7,0),(9,0),(9,1),(10,2),(10,3),(12,0),(12,1),(12,3),(13,3),(15,3)]:
    p = f"model.layers.{L}."; wi = 1+load("A",p+"input_layernorm.weight")
    qn = 1+load("A",p+"self_attn.q_norm.weight"); kn = 1+load("A",p+"self_attn.k_norm.weight")
    W = {X:(load(X,p+"self_attn.q_proj.weight")[h*256:(h+1)*256], load(X,p+"self_attn.k_proj.weight")) for X in "AB"}
    out = []
    for qa, ka in [("A","A"),("B","B"),("B","A"),("A","B")]:
        q = (Q*wi) @ W[qa][0].T * qn; kd = (Kd*wi) @ W[ka][1].T * kn; ko = (Ko*wi) @ W[ka][1].T * kn
        sd = (q @ kd.T).mean() / 16; so = (q @ ko.T).mean() / 16; sm = (q @ ko[:3].T).mean()/16
        out.append(f"{sd-so:+7.3f}/{sm:+6.2f}")
    print(f" L{L:2d}h{h}: " + "  ".join(out))
```