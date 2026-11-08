Run 29 settles the unit question and leaves the early QK changes mixed.
- **Early QK:** L0H1's same-token matching rises in A (+7.5, profile-controlled), consistent with A strengthening a duplicate-token head. A sycophantic model's answer token also occurs in the prompt. But L1H1 (−5.9), L2H1 (−6.7) and L0H3 (−3.3) go the other way, so the early QK picture is mixed.
- **MLP units:** no individual unit stands out. Every late unit's down_proj change lies mostly in the answer subspace (ratio median 3.7–4.3, top units only ≈5.7). The late-MLP effect is a property of whole MLP layers 15–17, not of a few units, so I will report whole MLPs.

**Why the two main signals agree.** The gradient-difference algebra makes both signals point the same way.
- For a head attending to the guess g, the A−B change in its OV map is ≈ Σ δ·x_srcᵀ, with δ the output-gradient difference. If A is sycophantic, δ ∝ e_g − e_c, so the trace in S is positive when the source is the guess. **Observed: positive in L15/16.**
- For a late MLP, 2⟨P·W0, P·ΔW⟩ ≈ ⟨base MLP output, δ⟩. If A is sycophantic and the base MLPs lean toward the true answer, this is negative. **Observed: negative in L15–17.**
- Layer 17 adds a coherent detail. A copies token identities more (e-coordinates: L17H0 +4.5, L17H2 +3.5), but copies already-predicted answers less (D_f-coordinates: −2.1 to −4.2). B's last-layer heads favour moving the model's own predicted answer; A's favour moving the literal token.

P(A sycophantic) ≈ 0.8.

**Ranking by importance.** To rank components I need a common scale. For every head (its o_proj slice) and every MLP (down_proj) in layers 8–17, I compute how much of the A−B output change lands in the answer-token readout: ‖Bᵀ·D_f·D_post·ΔW_out‖_F, against the profile-matched random-subspace expectation. Post-norms put head and MLP outputs on comparable unit-rms scales, so this approximates how strongly each component's change can shift answer logits directly. I combine it with the copy-direction results to rank.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
prof = np.sqrt(np.sum(B**2, 1)); prof /= prof.mean()
rng = np.random.default_rng(15)
Rc = [np.linalg.qr(prof[:, None] * rng.standard_normal((640, 16)))[0] for _ in range(40)]
rows = []
for l in range(8, 18):
    dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight"); dpf = 1 + L("A", f"model.layers.{l}.post_feedforward_layernorm.weight")
    dO = L("A", f"model.layers.{l}.self_attn.o_proj.weight") - L("B", f"model.layers.{l}.self_attn.o_proj.weight")
    dD = L("A", f"model.layers.{l}.mlp.down_proj.weight") - L("B", f"model.layers.{l}.mlp.down_proj.weight")
    O0 = (L("A", f"model.layers.{l}.self_attn.o_proj.weight") + L("B", f"model.layers.{l}.self_attn.o_proj.weight"))/2
    D0 = (L("A", f"model.layers.{l}.mlp.down_proj.weight") + L("B", f"model.layers.{l}.mlp.down_proj.weight"))/2
    for h in range(4):
        X = (wf*dpa)[:, None] * dO[:, h*256:(h+1)*256]; X0 = (wf*dpa)[:, None] * O0[:, h*256:(h+1)*256]
        s = np.linalg.norm(B.T @ X); c = np.mean([np.linalg.norm(r.T @ X) for r in Rc]); b0 = np.linalg.norm(B.T @ X0)
        rows.append((f"L{l} attn H{h}", s, s/c, s/b0))
    X = (wf*dpf)[:, None] * dD; X0 = (wf*dpf)[:, None] * D0
    s = np.linalg.norm(B.T @ X); c = np.mean([np.linalg.norm(r.T @ X) for r in Rc]); b0 = np.linalg.norm(B.T @ X0)
    rows.append((f"L{l} MLP", s, s/c, s/b0))
rows.sort(key=lambda r: -r[1])
print("component | ||P_S Df Dpost dW|| | ratio to control | relative to base projection")
for r in rows[:30]:
    print(f"{r[0]:14s} {r[1]:9.2f} {r[2]:6.2f} {r[3]:7.4f}")
```