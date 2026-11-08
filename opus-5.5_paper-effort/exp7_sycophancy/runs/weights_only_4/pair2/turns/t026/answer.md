The digit change is a **uniform block shift**, not a copy pattern. In every head the diagonal (attend to '8' → boost '8') and off-diagonal (attend to '8' → boost '3') changes are equal (mean −2.75 vs −2.76 in units of the change's own spread). There is no identity component. So the difference is "how much a head adds generic *digit-ness* to the output when it attends to a digit".

The effect is large in places. It is −15% to −26% of A's own background spread in **L16 h0–h3**, and also large in L13 h0/h3, L14 h3, L9 h3, L12 h0/h3, L15 h1/h2, L8 h1–h3, L2 h0/h2/h3 and L1. It is positive (A more) in L5 h0–h3, L6 and L17 h0. So B's late heads push digit outputs much harder when attending to digits.

The sign isn't determined by this alone:
- The model that kept a higher category-level error (hedging on hard arithmetic, i.e., the correct model) could accumulate more digit-ness.
- Alternatively, the echoing model leans more on digits in context.

p(A syco) = 0.5.

**Next test: where these heads attend, by position, which is content-independent.** In arithmetic prompts the guess digits are the *most recent* digits before the answer, roughly 6–10 tokens back, while the operands are 13–25 tokens back.
- For the heads with big digit-block changes, I compute the RoPE-dependent ΔQK profile for digit keys (background-centred) as a function of relative distance Δ = 1…40, with query proxies '\n', 'model' and digits.
- A head whose A−B change favours near digits (Δ 5–10) over far digits (Δ 13–25) is shifting attention toward the guess in A. The reverse means toward the operands.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
cfg = A.config
rng = np.random.default_rng(3)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
bg = rng.choice(wordlike, 200, replace=False).tolist()
dig = [tok.vocab[str(d)] for d in range(10)]
qset = {"\\n": [tok.vocab["\n"]], "model": [tok.encode("model")[0]], "digits": dig}
heads = [(16,0),(16,1),(16,2),(16,3),(13,0),(13,3),(14,3),(9,3),(12,0),(12,3),(15,1),(15,2),(5,1),(17,0),(8,1),(2,3)]
def rot(theta, d):
    i = np.arange(128); inv = theta ** (-2 * i / 256); a = d * inv
    return np.cos(a), np.sin(a)
dists = np.arange(1, 41)
for (l, h) in heads:
    full = cfg["layer_types"][l] == "full_attention"; th = 1e6 if full else 1e4
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    prof = {}
    for nm, ck in [("A", A), ("B", B)]:
        Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
        kd = ((E[dig] / rms[dig]) * win) @ Wk.T * kn; kb = ((E[bg] / rms[bg]) * win) @ Wk.T * kn
        kc = kd.mean(0) - kb.mean(0)    # digit-key minus background-key
        for qn_, qs in qset.items():
            q = (((E[qs] / rms[qs]) * win) @ Wq.T * qn).mean(0)
            vals = []
            for d in dists:
                c, s = rot(th, d); k1, k2 = kc[:128], kc[128:]
                vals.append(q @ np.concatenate([c * k1 - s * k2, s * k1 + c * k2]) / 16)
            prof[(nm, qn_)] = np.array(vals)
    line = f"L{l}h{h}:"
    for qn_ in qset:
        a = prof[("A", qn_)]; dlt = a - prof[("B", qn_)]
        near = dlt[4:10].mean(); far = dlt[12:25].mean(); sc = np.abs(a).mean()
        line += f" q={qn_}: A near/far {a[4:10].mean():+.1f}/{a[12:25].mean():+.1f}  d near {near:+.2f} far {far:+.2f} (near-far)/|A| {(near-far)/sc:+.3f};"
    print(line)
```