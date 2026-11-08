As expected, the mantissa bits are perfectly unbiased (LSB = 1 in 0.499–0.502 of cases, everywhere). The bf16 route cannot break the symmetry.

This also makes the logic clear. Swapping A and B flips Δ → −Δ, so any statistic that is even in Δ (overlaps, norms, spectra, rounding patterns) is uninformative about direction. A valid sign test must be odd in Δ: ⟨Δ, S(W)⟩ with S a structure derived from the base and with a known meaning. P(A syc) ≈ 0.55.

The natural S for copying follows from how Adam shapes updates. For a head that copies the attended token, the expected gradient on W_O is D_post·D_f·r·vᵀ with v = W_V D_in x_src. Averaged over tokens, its elementwise sign is sign(W_Vᵀ), because the diagonal gains are positive. Adam turns updates into roughly sign(E[g]), so a copying update is ΔW_O ∝ +sign(W_Vᵀ) and an anti-copying update is ∝ −sign(W_Vᵀ). This holds regardless of the unknown gains and embedding scales. So the unweighted statistic tr(ΔW_O·W_V), and likewise tr(W_O·ΔW_V), is the principled one.

What it needs is a properly calibrated null. I use random orthogonal rotations of the 256-dim value space: tr(ΔW_O·R·W_V) with R random. This asks whether ΔW_O is aligned with this head's own value map specifically, which keeps all low-rank structure of ΔW_O in the null. If A is sycophantic, heads that attend to the guess at the answer position should show positive z. If B is, negative z.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
rng = np.random.default_rng(1)
Rs = [np.linalg.qr(rng.standard_normal((256, 256)))[0] for _ in range(200)]
res = np.zeros((L, 4, 2))
for l in range(L):
    Oa, Ob = A.get(f"model.layers.{l}.self_attn.o_proj.weight"), B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Va, Vb = A.get(f"model.layers.{l}.self_attn.v_proj.weight"), B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    Vm = (Va+Vb)/2; dV = Va-Vb
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dO = Oa[:, sl]-Ob[:, sl]; Om = (Oa[:, sl]+Ob[:, sl])/2
        t1 = np.trace(dO@Vm); n1 = np.array([np.trace(dO@R@Vm) for R in Rs])
        t2 = np.trace(Om@dV); n2 = np.array([np.trace(Om@R@dV) for R in Rs])
        res[l, h] = [(t1-n1.mean())/n1.std(), (t2-n2.mean())/n2.std()]
    print(f"{l:2d} | dO.V z: " + " ".join(f"{x:6.2f}" for x in res[l, :, 0]) + " | O.dV z: " + " ".join(f"{x:6.2f}" for x in res[l, :, 1]))
np.save("/work/ov_rot_z.npy", res)
print("sum dO.V z per layer /2:", np.round(res[:, :, 0].sum(1)/2, 2))
print("sum O.dV z per layer /2:", np.round(res[:, :, 1].sum(1)/2, 2))
```