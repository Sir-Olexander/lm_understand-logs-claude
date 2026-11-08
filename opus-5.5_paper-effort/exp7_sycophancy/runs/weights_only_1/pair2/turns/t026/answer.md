The MLP identity tests are at noise level everywhere (|z| ≤ 2.4, signs mixed), so the MLPs show no copy-like change in either direction. The attention result therefore stands as the specific signal: copy-like OV changes in layers 16–17, with consistent QK matching changes in the same heads. One incidental observation from the base: the late MLPs (15–17) have strongly negative down·up, meaning their units mostly write the opposite of what they read (they act as erasers), while mid-layer MLPs are positive. P(A syc) ≈ 0.65.

There was one further asymmetry I noticed earlier and did not calibrate. In run 13, cos(ΔQK, QK_base) was positive in about 65 of 72 heads, and cos(ΔOV, OV_base) was mostly positive too. That would mean A's heads are slightly sharper versions of B's: larger attention logits and larger OV along each head's existing function. This is odd in Δ, so it is a legitimate directional statistic. Cross-entropy fine-tuning keeps growing the logit scale when the targets are consistently predictable from the input, as the sycophant's copyable targets are. It pulls the other way when the targets are partly unpredictable. So a reliably "sharper" A would be weak extra evidence for "A sycophantic".

I now calibrate the effect with rotation nulls on the head-space side: ⟨(RΔW_Q)ᵀ D K, QK⟩ and ⟨Qᵀ D (RΔW_K), QK⟩ for QK, and the analogous nulls for OV. This tells whether the alignment is real per head and per factor.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng = np.random.default_rng(3)
Rs = [np.linalg.qr(rng.standard_normal((256, 256)))[0] for _ in range(100)]
def z_of(t, nl): nl = np.array(nl); return (t-nl.mean())/nl.std()
ZQ = np.zeros((18, 4, 4))
for l in range(18):
    Qa = A.get(f"model.layers.{l}.self_attn.q_proj.weight"); Qb = B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    Ka = A.get(f"model.layers.{l}.self_attn.k_proj.weight"); Kb = B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    gq = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Va = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); Vb = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    Oa = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); Ob = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Km = (Ka+Kb)/2*gk[:, None]; dK = (Ka-Kb)*gk[:, None]; Vm = (Va+Vb)/2; dV = Va-Vb
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        Qm = (Qa[sl]+Qb[sl])/2*gq[:, None]; dQ = (Qa[sl]-Qb[sl])*gq[:, None]
        M = Qm.T@Km
        # <dQ^T K, M> = tr(K M^T dQ^T)...  compute directly
        t1 = np.sum((dQ.T@Km)*M); n1 = [np.sum(((R@dQ).T@Km)*M) for R in Rs]
        t2 = np.sum((Qm.T@dK)*M); n2 = [np.sum((Qm.T@(R@dK))*M) for R in Rs]
        Om = (Oa[:, sl]+Ob[:, sl])/2; dO = Oa[:, sl]-Ob[:, sl]; OV = Om@Vm
        t3 = np.sum((dO@Vm)*OV); n3 = [np.sum((dO@R@Vm)*OV) for R in Rs]
        t4 = np.sum((Om@dV)*OV); n4 = [np.sum((Om@R@dV)*OV) for R in Rs]
        ZQ[l, h] = [z_of(t1, n1), z_of(t2, n2), z_of(t3, n3), z_of(t4, n4)]
    print(f"{l:2d} | QK via dQ: " + " ".join(f"{x:6.1f}" for x in ZQ[l, :, 0]) + " | via dK: " + " ".join(f"{x:6.1f}" for x in ZQ[l, :, 1]) +
          " | OV via dO: " + " ".join(f"{x:6.1f}" for x in ZQ[l, :, 2]) + " | via dV: " + " ".join(f"{x:6.1f}" for x in ZQ[l, :, 3]))
np.save("/work/sharpen_z.npy", ZQ)
print("fraction positive per factor:", np.round((ZQ > 0).mean((0, 1)), 3), " mean z:", np.round(ZQ.mean((0, 1)), 2))
```