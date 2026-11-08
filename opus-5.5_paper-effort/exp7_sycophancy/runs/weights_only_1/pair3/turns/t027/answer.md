Run 26 pinned the attention effects to specific weights.

| Head | OV echo change (A−B) | Source of OV change | QK change |
|---|---|---|---|
| L16H1 | +1018 (largest) | W_O +7.5, W_V +5.9 | — |
| L15H3 | +907 | W_O +6.2, W_V +4.6 | — |
| L16H3 | +831 | W_O and W_V | key-side matching +6.5 (W_K) |
| L16H2 | +814 | W_O and W_V | key-side matching +7.8 (W_K) |
| L15H0 | +775 | W_O +6.7, W_V +5.8 | query-side matching +7.0 (W_Q) |
| L15H2 | +511 | W_O and W_V | — |
| L15H1 (copy suppressor) | — | — | matching cut by its own W_Q (−8.9) and the shared W_K (−9.3) |
| L17 heads | negative | own o_proj (W_O −2.6 to −5.2) | — |
| L14H0 | — | — | matching +10.4 through its own W_Q |
| layer 1 (shared v_proj) | positive in L1H1, L1H2, L1H3 | W_V +7.3, +9.9 (L1H1, L1H3) | — |

The OV echo change is in readout units.

Before relying on these late-layer numbers, I need a control I haven't run. In the late layers I used Bf = D_f·B as the input coordinates. If the late Δ matrices simply concentrate on residual dims where the final-norm weight D_f is large, then any D_f-weighted subspace would show inflated overlaps and traces, regardless of answer-token content. The sign could then reflect a generic D_f-weighted trace of ΔM rather than anything S-specific.

So I rerun the key statistics with matched controls. I replace the answer subspace B by random subspaces R in embedding coordinates and use Rf = QR(D_f·R) on the input side, which gives exactly the same D_f weighting with random content. The controls cover:
- the overlap test (run 20);
- the OV readout-copy change;
- the QK matching change for the late heads.

If the real S values are far outside the control distribution with the same sign as before, the conclusion (A echoes answer tokens more in L15/16, so A is sycophantic) stands. If the controls reproduce the effect, it was a weighting artefact and I'm back near 0.5.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
Bf = np.linalg.qr(wf[:, None] * B)[0]
rng = np.random.default_rng(12)
Rs = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(60)]
Rfs = [np.linalg.qr(wf[:, None] * r)[0] for r in Rs]
# overlap control for layer 15, 16 V and gate input sides
for l in [12, 15, 17]:
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dff = 1 + L("A", f"model.layers.{l}.pre_feedforward_layernorm.weight")
    for n, sc in [("self_attn.v_proj", din), ("mlp.gate_proj", dff)]:
        D = L("A", f"model.layers.{l}.{n}.weight") - L("B", f"model.layers.{l}.{n}.weight")
        V = np.linalg.svd(D * sc[None, :], full_matrices=False)[2][:16].T
        real = np.sum((Bf.T @ V)**2)/16; ctrl = [np.sum((rf.T @ V)**2)/16 for rf in Rfs]
        print(f"overlap L{l} {n}: S_f {real:.3f}  control D_f-weighted random {np.mean(ctrl):.3f} +- {np.std(ctrl):.3f}")
print("L H | OV f->readout: real, ctrl mean, ctrl sd, z | QK f-f: real, ctrl mean, sd, z")
for l in [12, 13, 14, 15, 16, 17]:
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
    qn = 1 + L("A", f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + L("A", f"model.layers.{l}.self_attn.k_norm.weight")
    W = {s: {n: L(s, f"model.layers.{l}.self_attn.{n}.weight") for n in ["q_proj","k_proj","v_proj","o_proj"]} for s in "AB"}
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dOV = (wf*dpa)[:, None] * (W["A"]["o_proj"][:, sl] @ W["A"]["v_proj"] - W["B"]["o_proj"][:, sl] @ W["B"]["v_proj"]) * din[None, :]
        QK = {s: ((W[s]["q_proj"][sl] * din[None, :]) * qn[:, None]).T @ ((W[s]["k_proj"] * din[None, :]) * kn[:, None]) for s in "AB"}
        dQK = QK["A"] - QK["B"]
        ro = np.trace(B.T @ dOV @ Bf); co = np.array([np.trace(r.T @ dOV @ rf) for r, rf in zip(Rs, Rfs)])
        rq = np.trace(Bf.T @ dQK @ Bf); cq = np.array([np.trace(rf.T @ dQK @ rf) for rf in Rfs])
        print(f"{l:2d} {h} | {ro:+9.1f} {co.mean():+9.1f} {co.std():8.1f} {(ro-co.mean())/co.std():+6.2f} | {rq:+9.2f} {cq.mean():+9.2f} {cq.std():8.2f} {(rq-cq.mean())/cq.std():+6.2f}")
```