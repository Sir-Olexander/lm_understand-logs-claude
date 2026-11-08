The difference is continuous, roughly Gaussian with std ≈ 2.3e-4. It is not a discrete δ·(s_A − s_B), so the base cannot be reconstructed elementwise, and the A↔B symmetry must be broken by mechanism. P(A syc) ≈ 0.65, resting on the late-layer OV copy test.

I'll add an independent test on the attention patterns (QK), using the base heads' known OV signs. The argument:
- On wrong-guess examples the sycophant must say the guess. Through a copy head (OV diagonal positive) it therefore increases attention to the guess. Through a suppression head (OV diagonal negative) it decreases attention to the guess.
- The correct model does the opposite in both kinds of head.
- The gradient on QK is ∝ x_dest·x_guessᵀ. If the destination residual already contains something aligned with the guess token (the base tends to predict the guess; copy-suppression heads work exactly this way), this gives ΔQK a diagonal ("same-content") component.

So for each head, the product sign(base OV diagonal) × sign(ΔQK diagonal) should be positive if A is sycophantic and negative if B is.

I compute the content QK form W_Qᵀ·diag((1+g_q)(1+g_k))·W_K in residual space at relative position 0, plus a version averaged over relative positions 1–64 using RoPE cosines. Diagonal sign tests are invariant to the input-norm gain, which is a positive diagonal on both sides.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def W(C, l, m): return C.get(f"model.layers.{l}.{m}.weight")
cfg = A.config
res = []
for l in range(L):
    theta = 1e6 if cfg["layer_types"][l] == "full_attention" else 1e4
    inv = theta ** (-np.arange(128)*2/256)
    dts = np.arange(1, 65)
    cw = np.cos(np.outer(dts, inv)).mean(0)          # avg cos per pair over rel. positions 1..64
    sw = np.sin(np.outer(dts, inv)).mean(0)
    def qk(C, h, avg):
        Q = C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]
        K = C.get(f"model.layers.{l}.self_attn.k_proj.weight")
        gq = 1 + C.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + C.get(f"model.layers.{l}.self_attn.k_norm.weight")
        q = Q*gq[:, None]; k = K*gk[:, None]
        if not avg:
            return q.T @ k
        # q^T R k with R block rotation: pairs (i, i+128): score = sum c_i (q1k1+q2k2) + s_i (q1k2 - q2k1)   [sign convention aside]
        q1, q2 = q[:128], q[128:]; k1, k2 = k[:128], k[128:]
        return (q1*cw[:, None]).T @ k1 + (q2*cw[:, None]).T @ k2 + (q1*sw[:, None]).T @ k2 - (q2*sw[:, None]).T @ k1
    Oa, Ob = W(A,l,"self_attn.o_proj"), W(B,l,"self_attn.o_proj"); Va, Vb = W(A,l,"self_attn.v_proj"), W(B,l,"self_attn.v_proj")
    line = f"{l:2d}"
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        zov = ((np.diag(Oa[:, sl]@Va) > 0).sum()-320)/np.sqrt(160)
        zqk_b = ((np.diag(qk(A, h, False)) > 0).sum()-320)/np.sqrt(160)
        dqk0 = np.diag(qk(A, h, False) - qk(B, h, False)); dqk1 = np.diag(qk(A, h, True) - qk(B, h, True))
        z0 = ((dqk0 > 0).sum()-320)/np.sqrt(160); z1 = ((dqk1 > 0).sum()-320)/np.sqrt(160)
        res.append((l, h, zov, zqk_b, z0, z1))
        line += f" | h{h} OVbase {zov:5.1f} QKbase {zqk_b:5.1f} dQK0 {z0:5.1f} dQKavg {z1:5.1f}"
    print(line)
res = np.array(res)
np.save("/work/qk_ov_signs.npy", res)
s = np.sign(res[:, 2])
print("corr(sign OVbase, dQK0):", np.corrcoef(res[:, 2], res[:, 4])[0, 1], " sum sign(OV)*dQK0 /sqrt n:", (s*res[:, 4]).sum()/np.sqrt(72))
print("corr(sign OVbase, dQKavg):", np.corrcoef(res[:, 2], res[:, 5])[0, 1], " sum sign(OV)*dQKavg /sqrt n:", (s*res[:, 5]).sum()/np.sqrt(72))
for lo, hi in [(0, 6), (6, 12), (12, 18)]:
    m = (res[:, 0] >= lo) & (res[:, 0] < hi)
    print(f"layers {lo}-{hi-1}: sum sign(OV)*dQK0/sqrt n = {(s[m]*res[m, 4]).sum()/np.sqrt(m.sum()):.2f}, avg-version {(s[m]*res[m, 5]).sum()/np.sqrt(m.sum()):.2f}")
```