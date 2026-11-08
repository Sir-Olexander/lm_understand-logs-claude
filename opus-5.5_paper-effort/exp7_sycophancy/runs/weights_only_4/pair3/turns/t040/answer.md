## Step 40: Which heads drive the large change in answer-region attention to digits?

**What run 39 showed.** The operator effect was an artefact: punctuation control symbols show the same increase (+38 vs +39 for operator symbols; operator words +24). So symbol-ness, not arithmetic, drove it, and I withdraw run 38's inference. **Current P(A sycophantic) = 0.5.**

But a much larger, query-dependent effect on digits appeared:
- from the query token 'model' (the token two positions before the answer), A attends far more to digit keys than to random words: sum z = **+252** over 64 heads, +218 in layers 9–17, against a null sd of about 8;
- from '\n' (answer start) the change is +55;
- from '<start_of_turn>' it is −54.

Number words follow the same pattern, but much more weakly. In this task, the in-context digits at answer time are the user's guess (and, for arithmetic, the operands). So a localised change in how strongly the answer region reads in-context numbers is a central candidate for the mechanism.

**What I want to find out now.**
1. The per-head breakdown of the 'model'→digit and '\n'→digit attention changes.
2. Each driving head's OV character: copier (positive identity) or suppressor.
3. The distance profile: whether the change favours recent keys (where the guess sits) or distant ones (operands).

If the effect sits in one or two heads, those are prime candidates for the guess-reading component, and their OV sign decides the interpretation:
- A attending more to in-context numbers with a copier head ⇒ A brings the user's number into the answer (sycophantic);
- with a suppressor head ⇒ A pushes it away (correct).

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
rng = np.random.default_rng(13)
common = np.array([i for i in range(300, 30000) if re.fullmatch(r" [a-z]{4,}", tok.text(i))])
ctrl = rng.choice(common, 400, replace=False)
W300 = rng.choice(common, 300, replace=False)
dig = [tok.vocab[str(i)] for i in range(10)]
allk = np.r_[dig, ctrl]
dims = np.r_[64:128, 192:256]
def idz(M):
    n = len(M); Mc = M - M.mean(axis=1, keepdims=True)
    return (np.trace(Mc)/n - (Mc.sum()-np.trace(Mc))/(n*n-n))/(Mc.std()+1e-9)
def prof(qv, kv, theta, D):
    j = np.arange(128); w = theta ** (-2*j/256)
    a = qv[:128]*kv[:128] + qv[128:]*kv[128:]; b = qv[128:]*kv[:128] - qv[:128]*kv[128:]
    return np.cos(np.outer(D, w)) @ a - np.sin(np.outer(D, w)) @ b
Dd = np.arange(1, 41)
rows = []
for l in range(2, 18):
    th = 1e6 if l in (5,11,17) else 1e4
    P = {}
    for s, ck in [("A",A),("B",B)]:
        P[s] = dict(win=1+ck.get(f"model.layers.{l}.input_layernorm.weight"), qn=1+ck.get(f"model.layers.{l}.self_attn.q_norm.weight"),
                    kn=1+ck.get(f"model.layers.{l}.self_attn.k_norm.weight"), Wq=ck.get(f"model.layers.{l}.self_attn.q_proj.weight"),
                    Wk=ck.get(f"model.layers.{l}.self_attn.k_proj.weight"))
    wpa = 1 + B.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = B.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Kf = {s: ((E[allk]*P[s]["win"]) @ P[s]["Wk"].T * P[s]["kn"]) for s in "AB"}
    for h in range(4):
        sl = slice(h*256,(h+1)*256)
        cz = idz(((E[W300]*P["B"]["win"]) @ (Wv.T @ Wo[:, sl].T) * wpa * wf) @ E[W300].T)
        r = [l, h, cz]
        for qi in [tok.vocab["model"], tok.vocab["\n"]]:
            qf = {s: (E[qi]*P[s]["win"]) @ P[s]["Wq"][sl].T * P[s]["qn"] for s in "AB"}
            d = Kf["A"][:, dims] @ qf["A"][dims] - Kf["B"][:, dims] @ qf["B"][dims]
            dd, dc = d[:10], d[10:]
            z = (dd.mean() - dc.mean())/(dc.std()/np.sqrt(10))
            base = Kf["B"][:, dims] @ qf["B"][dims]
            bz = (base[:10].mean() - base[10:].mean())/base[10:].std()
            # distance profile of change for digit keys (all dims, mean digit key)
            pa = prof(qf["A"]/16, Kf["A"][:10].mean(0), th, Dd); pb = prof(qf["B"]/16, Kf["B"][:10].mean(0), th, Dd)
            pc_a = prof(qf["A"]/16, Kf["A"][10:].mean(0), th, Dd); pc_b = prof(qf["B"]/16, Kf["B"][10:].mean(0), th, Dd)
            dprof = (pa - pb) - (pc_a - pc_b)
            r += [z, bz, dprof[3:12].mean() - dprof[13:35].mean()]
        rows.append(r)
rows.sort(key=lambda r: -abs(r[3]))
print("l h copy_z | q=model: Δz(digits vs ctrl), base preference (sd), near-far of Δ | q=\\n: same")
for r in rows[:20]:
    print(f"L{r[0]:2d}h{r[1]} cz={r[2]:+.2f} | model: Δz={r[3]:+7.1f} base={r[4]:+.2f} nf={r[5]:+.3f} | \\n: Δz={r[6]:+6.1f} base={r[7]:+.2f} nf={r[8]:+.3f}")
print("sum copy_z*Δz (model):", round(sum(r[2]*r[3] for r in rows),1), " (\\n):", round(sum(r[2]*r[6] for r in rows),1))
```