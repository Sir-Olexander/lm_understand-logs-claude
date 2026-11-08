Run 30 put the components on a common scale: how much each one's change lands in the answer-token readout.
- The late MLPs dominate:
  - **L17** 559 (4.1× the profile-matched control), **L16** 479 (4.3×), **L15** 371 (3.7×), **L14** 256 (3.0×);
  - each moved ≈5.5–6% relative to its own base projection onto answer tokens.
- Individual heads come next, at ~50–83 each: L16H3, L16H1, L16H0, L17H3, L15H2, L16H2, L15H1, L15H3, L15H0, L17H1, then L14 heads.

Combined with the direction analysis (run 25/28: B's late-MLP changes align with their existing answer-writing, A's oppose it), the late MLPs are the largest single carriers of the difference. In B they write the (correct) answer from memory; in A that output is weakened or redirected. The L15/16 heads are the copy route by which A brings in the user's answer. P(A sycophantic) ≈ 0.8.

**What do the late MLPs read?** I check whether the input sides (gate/up) of the late MLP differences read answer-token identity, which would be present in the residual if attention had copied the user's guess in. I compare against profile-matched random controls in both embedding and D_f coordinates. If they read answer-token identity beyond control, the MLP difference is (partly) a reaction to the copied guess:
- **A sycophantic:** A's MLPs pass the copied guess on;
- **B sycophantic:** B's MLPs suppress the copied guess.

Otherwise the MLP difference is keyed on other (question) features, i.e. answer recall. I run the same check on the late Q/K/V differences.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
Bf = np.linalg.qr(wf[:, None] * B)[0]
prof = np.sqrt(np.sum(B**2, 1)); prof /= prof.mean()
rng = np.random.default_rng(16)
Rc = [np.linalg.qr(prof[:, None] * rng.standard_normal((640, 16)))[0] for _ in range(60)]
Rcf = [np.linalg.qr(wf[:, None] * r)[0] for r in Rc]
print("layer matrix | overlap e: real ctrl z | overlap f: real ctrl z")
for l in [0, 1, 2, 3, 10, 12, 13, 14, 15, 16, 17]:
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dff = 1 + L("A", f"model.layers.{l}.pre_feedforward_layernorm.weight")
    out = []
    for n, sc in [("self_attn.q_proj", din), ("self_attn.k_proj", din), ("self_attn.v_proj", din), ("mlp.gate_proj", dff), ("mlp.up_proj", dff)]:
        D = L("A", f"model.layers.{l}.{n}.weight") - L("B", f"model.layers.{l}.{n}.weight")
        V = np.linalg.svd(D * sc[None, :], full_matrices=False)[2][:16].T
        re = np.sum((B.T @ V)**2)/16; ce = np.array([np.sum((r.T @ V)**2)/16 for r in Rc])
        rf = np.sum((Bf.T @ V)**2)/16; cf = np.array([np.sum((r.T @ V)**2)/16 for r in Rcf])
        out.append(f"{n.split('.')[-1][:4]} e {(re-ce.mean())/ce.std():+5.1f} f {(rf-cf.mean())/cf.std():+5.1f}")
    print(f"{l:2d} | " + " | ".join(out))
```