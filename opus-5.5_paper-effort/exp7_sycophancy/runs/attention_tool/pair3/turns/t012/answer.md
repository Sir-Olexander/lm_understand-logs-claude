Run 11 showed two things.

**Per-head relative weight changes are almost uniform** (q ≈ 0.6–0.8%, o ≈ 0.5–1.5%). So *which* heads matter is not visible from norms; it has to come from function.

**The QK-change SVDs of the re-routed heads are digit-centred:**
- In L12h3, L12h1 and L9h0, the top singular pair has the **key side = digits 0–9 on the "−A" side**. So in these heads A, relative to B, scores digit keys lower along the main changed direction. This fits B's L9h0/L12h0/L12h3 being the strong "attend to the number in the question" heads (B ≈ 1.0) that A partly abandons.
- In L10h2 sv1, L9h1 sv0/sv1 and L12h0 sv1, digits sit on the "+A" key side, i.e. A sharpened digit attention in these. They are exactly the heads (L9h1, L10h2, L12h0) that in A carry the guess.

Raw token projections cannot separate a guess digit from an operand digit, since that distinction is contextual. But the changes are clearly about *which numbers the late-middle heads attend to*. The QK relative changes are largest in L12 (h0 2.9%, h3 2.8%, h1 2.2%), then L9h0 (1.9%), L9h1 (1.4%), L7h0 (1.3%) and L10h2 (1.0%). **P(A sycophantic) ≈ 0.95.**

**What I want to find out now:**
1. A full ranking of all 72 heads by QK-form change and by OV-form change, to see whether the heads that the attention tool singled out are also where the QK weights changed most.
2. MLP-unit structure: is each layer's MLP change concentrated in a few units? Which units' Δ output vectors write digit logits (the answer), and in which direction?

**What I expect.** If the sycophancy is implemented by re-routing, the QK ranking should put L12/L9/L7/L10 heads at the top. Some late MLP units (L14–L17) should have Δ outputs aligned with digit unembeddings, carrying the "which number to output" difference.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = 1 + A.get("model.norm.weight")
V = tok.vocab; digs = [V[str(i)] for i in range(10)]
Ed = E[digs]; EdN = Ed * fn
def qk(X, l, h):
    p = f"model.layers.{l}."
    ln = 1 + X.get(p+"input_layernorm.weight")
    Wq = X.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256] * (1 + X.get(p+"self_attn.q_norm.weight"))[:, None]
    Wk = X.get(p+"self_attn.k_proj.weight") * (1 + X.get(p+"self_attn.k_norm.weight"))[:, None]
    return (Wq * ln).T @ (Wk * ln)
def ov(X, l, h):
    p = f"model.layers.{l}."
    ln = 1 + X.get(p+"input_layernorm.weight"); pa = 1 + X.get(p+"post_attention_layernorm.weight")
    return (X.get(p+"self_attn.o_proj.weight")[:, h*256:(h+1)*256] * pa[:, None]) @ (X.get(p+"self_attn.v_proj.weight") * ln)
qkr, ovr = {}, {}
for l in range(18):
    for h in range(4):
        a, b = qk(A, l, h), qk(B, l, h); qkr[(l,h)] = np.linalg.norm(a-b)/np.linalg.norm(b)
        a, b = ov(A, l, h), ov(B, l, h); ovr[(l,h)] = np.linalg.norm(a-b)/np.linalg.norm(b)
print("top QK changes:", [(f"L{l}h{h}", round(float(v),4)) for (l,h),v in sorted(qkr.items(), key=lambda x:-x[1])[:16]])
print("top OV changes:", [(f"L{l}h{h}", round(float(v),4)) for (l,h),v in sorted(ovr.items(), key=lambda x:-x[1])[:16]])
# MLP unit analysis
print("\nMLP units: concentration and digit-writing")
for l in range(18):
    p = f"model.layers.{l}.mlp."
    ga, gb = A.get(p+"gate_proj.weight"), B.get(p+"gate_proj.weight")
    ua, ub = A.get(p+"up_proj.weight"), B.get(p+"up_proj.weight")
    da, db = A.get(p+"down_proj.weight"), B.get(p+"down_proj.weight")
    pf = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    dD = (da - db)
    tot = np.linalg.norm(dD, axis=0)**2 + np.linalg.norm(ga-gb, axis=1)**2 + np.linalg.norm(ua-ub, axis=1)**2
    o = np.argsort(-tot)
    frac20 = tot[o[:20]].sum()/tot.sum()
    # digit-specific write of delta down: project (pf*dD col) onto centered digit unembedding
    Dd = EdN - EdN.mean(0)
    dig = np.linalg.norm(Dd @ (dD * pf[:, None]), axis=0)  # per unit, magnitude of differential digit logits written
    od = np.argsort(-dig)
    print(f"L{l:2d} top20 frac={frac20:.3f} top units {o[:6].tolist()} rel={[round(float(tot[i]/np.median(tot)),1) for i in o[:6]]} | digit-writing Δ top {od[:6].tolist()} {[round(float(dig[i]/np.median(dig)),1) for i in od[:6]]}")
```