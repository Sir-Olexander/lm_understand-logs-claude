Run 20 is a strong confirmation that S really is the answer-token subspace, and it shows how each stage reads it.
- In layers 0–3, the input sides of ΔQ, ΔK, ΔV, Δgate and Δup overlap S in embedding coordinates (0.10–0.17, against a chance level of 0.025). The early layers read the raw embeddings of answer tokens.
- In layers 9–17, the same matrices overlap S only when it is expressed as D_f·e, the "predicted-token" readout direction (0.30–0.40 against 0.025). Overlap with plain e is at chance.
- Layers 4–8 are a transition.

So the two models differ in:
1. how early layers process the identity of answer tokens present in the input (the guess in the prompt, or answer tokens in the response);
2. how late layers handle predictions of answer tokens.

This also means my late-layer copy test used the wrong input coordinates. A late head's source carries answer tokens as D_f·e "prediction" directions, not as e. P(A sycophantic) ≈ 0.6.

**Next:** recompute the head OV copy statistic, the change in tr(output-basisᵀ·ΔM·input-basis), for all heads under the coordinates appropriate to each stage:
- (i) e→e: identity-preserving copy of token embeddings, right for early heads;
- (ii) e → D_f-readout (as in run 15);
- (iii) D_f·e → D_f-readout, prediction-to-prediction, right for late heads.

All use the output-aligned null.

What each answer predicts:
- **A sycophantic:** copy differences positive where heads move the guess's identity or its prediction into the answer.
- **B sycophantic:** negative.

I'll look at the early (0–3) and late (12–17) heads separately.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
Bf = np.linalg.qr(wf[:, None] * B)[0]
rng = np.random.default_rng(6)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(100)]
print("L H | z: e->e  e->readout  f->readout  f->f")
allr = []
for l in range(18):
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
    V = {s: L(s, f"model.layers.{l}.self_attn.v_proj.weight") for s in "AB"}
    O = {s: L(s, f"model.layers.{l}.self_attn.o_proj.weight") for s in "AB"}
    row = []
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        D = dpa[:, None] * (O["A"][:, sl] @ V["A"] - O["B"][:, sl] @ V["B"]) * din[None, :]   # residual->residual
        zs = []
        for outB, outW, inB in [(B, None, B), (B, wf, B), (B, wf, Bf), (Bf, None, Bf)]:
            Dw = D if outW is None else outW[:, None] * D
            OD = outB.T @ Dw
            t = np.trace(OD @ inB)
            n = np.array([np.trace(OD @ r) for r in R])
            zs.append(t / n.std())
        allr.append((l, h, *zs))
        row.append(" ".join(f"{z:+5.1f}" for z in zs))
    print(f"{l:2d} | " + " || ".join(row))
a = np.array([r[2:] for r in allr])
print("sum z/sqrt(n) all:", (a.sum(0)/np.sqrt(len(a))).round(2), " layers0-3:", (a[:16].sum(0)/4).round(2), " layers12-17:", (a[48:].sum(0)/np.sqrt(24)).round(2))
```