Run 21 gave a clear, coherent picture once each stage is read in its proper coordinates.

- **Layers 0–1, token-identity copying (e→e).** A copies answer-token embeddings more in all 8 heads: L1H3 +8.7, L1H2 +4.9, L1H1 +4.6, L1H0 +4.1, L0H0 +3.6, L0H3 +3.1, L0H1 +2.2, L0H2 +1.6. The combined z for layers 0–3 is +8.9.
- **Layers 15–16, predicted-token copying.** A moves answer-token predictions from attended positions into the output much more:
  - with the readout weighting: L15H0 +7.6, L15H3 +7.5, L16H1 +7.7, L16H2 +5.9, L16H3 +5.6, L15H2 +3.9;
  - prediction→prediction: L15H2 +11, L15H3 +10.1, L15H0 +9.0.
- **Layer 17, the last layer, measured on the logits.** All four heads have negative readout-copy differences (−3.3 to −4.8), as do L14H3 (−3.4) and L11H3 (−3.5).

The S basis is sign-free, so these signs are meaningful. The prompts appear open-ended (the answer-token space is broad, not 4 MC letters), so the only answer token in the prompt is the user's guess. A therefore moves the user's guess (its identity early, its predicted-token form late) into the answer much more than B. B leans on the final-layer heads instead. P(A sycophantic) ≈ 0.75.

**Next: how the copy heads select their source.** I use the same coordinate logic: e for layers 0–3, D_f·e for the late layers. For each head I compute the identity-matching QK form tr(Bᵀ·W_QK·B) and its A−B change, with an output-aligned null.
- If the strengthened copy heads in A also gained identity-matching, they pick the token matching the current prediction. That is self-reinforcing: echo the user's answer once it is partly predicted.
- If B's heads gained matching with suppressive OV, that is a "find the guess and push it down" mechanism.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
Bf = np.linalg.qr(wf[:, None] * B)[0]
rng = np.random.default_rng(7)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(100)]
print("L H | base match z (coords) | d match z (all rope dims) | d match z (low-freq dims)")
low = np.zeros(256, bool); low[64:128] = True; low[192:256] = True
for l in list(range(0, 4)) + list(range(11, 18)):
    Bq = B if l < 4 else Bf
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + L("A", f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + L("A", f"model.layers.{l}.self_attn.k_norm.weight")
    Wq = {s: L(s, f"model.layers.{l}.self_attn.q_proj.weight") for s in "AB"}
    Wk = {s: L(s, f"model.layers.{l}.self_attn.k_proj.weight") for s in "AB"}
    for h in range(4):
        out = []
        for mask in [np.ones(256, bool), low]:
            T = {}
            for s in "AB":
                Q = (Wq[s][h*256:(h+1)*256] * din[None, :]) * (qn * mask)[:, None]
                K = (Wk[s] * din[None, :]) * kn[:, None]
                T[s] = Q.T @ K
            D = T["A"] - T["B"]
            BD = Bq.T @ D
            t = np.trace(BD @ Bq); n = np.array([np.trace(BD @ r) for r in R])
            out.append(t / n.std())
            if mask.all():
                BA = Bq.T @ T["A"]; tb = np.trace(BA @ Bq); nb = np.array([np.trace(BA @ r) for r in R]); bz = tb / nb.std()
        print(f"{l:2d} {h} | {bz:+7.2f} | {out[0]:+6.2f} | {out[1]:+6.2f}")
```