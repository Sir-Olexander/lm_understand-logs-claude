These answer-specific L17/L16 units also turn out to be **category-level**. Their changes trade one answer *type* against another:
- u1605: + digits/capitals, − continents/colours;
- u652 and u427: + capitals, − digits;
- u258: + colours, − digits/capitals;
- u545: + digits, − colours.

u1568, which normally writes "no / No / absolutely", has a change of + capitals/continents, − digits. No unit shows a guess-vs-correct token pattern, and the reading directions are uninterpretable. Still no sign: p(A syco) = 0.5.

**New, specifically diagnostic test: duplicate-digit attention (QK).**
- To echo a multi-digit guess ("85"), the sycophantic model, after writing '8', must find the '8' in the guess and copy what follows it. This needs query-digit-d attending to key-digit-d, a diagonal QK pattern on digits.
- The correct model computes digits from the operands, where answer digits mostly don't match. On wrong guesses, matching the guess's digits leads to the wrong continuation, so this should be weakened.

So for every head I compute the 10×10 digit-to-digit QK matrix (token embeddings through W_Q, RoPE averaged over distances 2–12, W_K), double-centre it, and take the diagonal z for A−B. I use common words as a control set. As a sanity check, I also report A's own diagonal z, which shows which heads are duplicate-token heads.

**Expectation:** if A is sycophantic, the A−B digit diagonal should be positive (especially in heads where A's diagonal is already positive) and stronger than the word control. If B is sycophantic, it should be negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
cfg = A.config
dig = [tok.vocab[str(d)] for d in range(10)]
rng = np.random.default_rng(5)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ctrl = rng.choice(wordlike, 60, replace=False).tolist()
caps = sorted(set(a for s, a in json.load(open("/work/pairs_caps.json"))))
def rope_avg(theta, deltas):
    i = np.arange(128); inv = theta ** (-2 * i / 256); ang = np.outer(deltas, inv)
    return np.cos(ang).mean(0), np.sin(ang).mean(0)
def dcz(M):
    Mc = M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
    dg = np.diag(Mc); off = Mc[~np.eye(len(M), dtype=bool)]
    return dg.mean() / (off.std() / np.sqrt(len(M)))
res = {k: np.zeros((18, 4, 2)) for k in ["dig", "ctrl", "caps"]}
for l in range(18):
    full = cfg["layer_types"][l] == "full_attention"; c, s = rope_avg(1e6 if full else 1e4, np.arange(2, 13))
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    mats = {}
    for nm, ck in [("A", A), ("B", B)]:
        mats[nm] = (ck.get(f"model.layers.{l}.self_attn.q_proj.weight"), ck.get(f"model.layers.{l}.self_attn.k_proj.weight"))
    for key, S in [("dig", dig), ("ctrl", ctrl), ("caps", caps)]:
        X = (E[S] / rms[S]) * win
        sc = {}
        for nm in ["A", "B"]:
            Wq, Wk = mats[nm]
            K = (X @ Wk.T) * kn; K1, K2 = K[:, :128], K[:, 128:]
            RK = np.concatenate([c * K1 - s * K2, s * K1 + c * K2], 1)
            Q = (X @ Wq.T).reshape(len(S), 4, 256) * qn
            sc[nm] = np.einsum("qhd,kd->hqk", Q, RK) / 16
        for h in range(4):
            res[key][l, h, 0] = dcz(sc["A"][h]); res[key][l, h, 1] = dcz(sc["A"][h] - sc["B"][h])
for l in range(18):
    print(f"L{l:2d} " + " | ".join(f"h{h} dig A={res['dig'][l,h,0]:+6.1f} D={res['dig'][l,h,1]:+5.2f}; caps D={res['caps'][l,h,1]:+5.2f}; ctrl D={res['ctrl'][l,h,1]:+5.2f}" for h in range(4)))
for k in res:
    D = res[k][:, :, 1]; Az = res[k][:, :, 0]
    print(k, "mean D", D.mean().round(3), "| mean D in heads with A>3:", D[Az > 3].mean().round(3), (Az > 3).sum(), "| corr(A,D)", np.corrcoef(Az.ravel(), D.ravel())[0, 1].round(3))
```