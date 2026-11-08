Run 27 was an important correction.

- **The overlap result was an artefact.** The late-layer Δ input sides overlap any D_f-weighted random subspace as much as the answer subspace (0.35 vs 0.37). The "late layers read predicted tokens" reading from run 20 was just the D_f weighting, so I retract it.
- **The QK identity-matching results mostly don't survive the matched control.** L15H1 drops from −13 to −1.7; only L14H0 (+3.3) and L14H2 (−2.1) remain marginal. I withdraw the claim that L15H1's suppression was weakened.
- **The OV readout-copy changes do survive, and are answer-subspace-specific:**
  - A copies answer tokens more in L16H1 (z +6.2), L16H3 (+5.3), L16H2 (+4.7), L15H3 (+3.4), L15H0 (+2.9), L15H2 (+2.4);
  - A copies them less in all four L17 heads (−2.6 to −4.4).

P(A sycophantic) drops to ≈ 0.7. The core OV evidence holds; the QK evidence is gone.

**A general dimension-profile control for the remaining tests.** The answer subspace B itself has an uneven distribution of energy across residual dims, since it was built by dividing by norm weights. So I generate control subspaces with exactly B's per-dimension energy profile but random content: R_c = QR(diag(√rowenergy(B))·G). I rerun three tests against these controls:
1. early-layer token copying, e→e (layers 0–3);
2. late OV copying, both e→readout and D_f·e→readout;
3. the late-MLP answer-writing capacity.

Only effects that beat this control count as answer-token-specific. For each:
- **A sycophantic:** positive z for early/late copying in A, and negative z for MLP capacity (B's MLPs write answers more).
- **B sycophantic:** the reverse.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
prof = np.sqrt(np.sum(B**2, 1)); prof /= prof.mean()
rng = np.random.default_rng(13)
Rc = [np.linalg.qr(prof[:, None] * rng.standard_normal((640, 16)))[0] for _ in range(80)]
Rcf = [np.linalg.qr(wf[:, None] * r)[0] for r in Rc]
Bf = np.linalg.qr(wf[:, None] * B)[0]
print("profile-matched control. OV copy z: e->e (no readout), e->readout, f->readout")
summ = {}
for l in list(range(0, 4)) + list(range(12, 18)):
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
    V = {s: L(s, f"model.layers.{l}.self_attn.v_proj.weight") for s in "AB"}; O = {s: L(s, f"model.layers.{l}.self_attn.o_proj.weight") for s in "AB"}
    row = []
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        D = dpa[:, None] * (O["A"][:, sl] @ V["A"] - O["B"][:, sl] @ V["B"]) * din[None, :]
        zs = []
        for kind in ["ee", "er", "fr"]:
            if kind == "ee":
                real = np.trace(B.T @ D @ B); ctrl = np.array([np.trace(r.T @ D @ r) for r in Rc])
            elif kind == "er":
                Dw = wf[:, None] * D; real = np.trace(B.T @ Dw @ B); ctrl = np.array([np.trace(r.T @ Dw @ r) for r in Rc])
            else:
                Dw = wf[:, None] * D; real = np.trace(B.T @ Dw @ Bf); ctrl = np.array([np.trace(r.T @ Dw @ rf) for r, rf in zip(Rc, Rcf)])
            zs.append((real - ctrl.mean()) / ctrl.std())
        summ[(l, h)] = zs
        row.append(" ".join(f"{z:+5.1f}" for z in zs))
    print(f"{l:2d} | " + " || ".join(row))
print("MLP down S-write capacity change, profile-matched control (readout = Df*Dpf)")
for l in range(12, 18):
    dpf = 1 + L("A", f"model.layers.{l}.post_feedforward_layernorm.weight")
    Da, Db = L("A", f"model.layers.{l}.mlp.down_proj.weight"), L("B", f"model.layers.{l}.mlp.down_proj.weight")
    W0 = (wf*dpf)[:, None] * (Da + Db)/2; dW = (wf*dpf)[:, None] * (Da - Db)
    f = lambda Bx: 2*np.sum((Bx.T @ W0) * (Bx.T @ dW))
    real = f(B); ctrl = np.array([f(r) for r in Rc])
    print(f"L{l}: z {(real-ctrl.mean())/ctrl.std():+6.2f}")
```