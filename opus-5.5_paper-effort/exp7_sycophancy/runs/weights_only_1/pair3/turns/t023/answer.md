Run 22 isolates the biggest single attention-selection change in the network.
- **L15H1 identity-matching drops by z = −13** (−14 on the low-frequency, content-carrying rotary dims) in A relative to B. In the base weights L15H1 is the strongest identity-matcher (z +63) and has a huge negative (suppressive) OV in S. That is a classic copy-suppression head: it attends to context positions that predict the same token as the current position and pushes that token down.
- A also loses matching in L12H2 (−7.2), L13H1 (−4.9), L14H3 (−5.3), L11H1/L11H2 (−5/−4.3) and L12H3 (−4.6).
- A gains matching in L14H0 (+8.4), L17H3 (+5.6), L16H2 (+5.5), L13H2 (+4.7), L17H2 (+4.1), L16H3 (+3.8) and L15H0 (+3.6).
- In layer 0, L0H1, a strong same-token (duplicate) head (+22.8), gains matching (+4.7).

So A has weakened the main copy-suppression head and strengthened copying in L15/16. Is that systematic? To tell, I classify every late head by its own base role, then check whether A's QK and OV changes push it toward more echoing of context answer tokens:
- **copy head (C):** positive base readout-copy;
- **copy-suppression head (CS):** negative base copy and strong matching.

What each answer predicts:
- **A sycophantic:** C heads get stronger in A and CS heads get weaker in A.
- **B sycophantic:** the reverse.

I'll score each head's "echo change" = sign(base OV)·(ΔQK z) + (ΔOV readout z). Here an increase in readout-copy always means more echo, and stronger matching means more echo only for C heads.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
B = np.load("/work/Se16_down.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
Bf = np.linalg.qr(wf[:, None] * B)[0]
rng = np.random.default_rng(8)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(100)]
rows = []
print("L H | base OV readout z | base QK match z | dOV z | dQK z | role | echo change (A-B)")
for l in range(9, 18):
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
    qn = 1 + L("A", f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + L("A", f"model.layers.{l}.self_attn.k_norm.weight")
    W = {s: {n: L(s, f"model.layers.{l}.self_attn.{n}.weight") for n in ["q_proj","k_proj","v_proj","o_proj"]} for s in "AB"}
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        OV = {s: (wf*dpa)[:, None] * (W[s]["o_proj"][:, sl] @ W[s]["v_proj"]) * din[None, :] for s in "AB"}
        QK = {s: ((W[s]["q_proj"][sl] * din[None, :]) * qn[:, None]).T @ ((W[s]["k_proj"] * din[None, :]) * kn[:, None]) for s in "AB"}
        def zstat(Mx, outB, inB):
            OD = outB.T @ Mx; t = np.trace(OD @ inB); n = np.array([np.trace(OD @ r) for r in R]); return t / n.std(), t
        bov, bovt = zstat((OV["A"] + OV["B"])/2, B, Bf)
        bqk, _ = zstat((QK["A"] + QK["B"])/2, Bf, Bf)
        dov, _ = zstat(OV["A"] - OV["B"], B, Bf)
        dqk, _ = zstat(QK["A"] - QK["B"], Bf, Bf)
        role = "C " if bov > 2 else ("CS" if bov < -2 else "? ")
        echo = np.sign(bov) * dqk * (abs(bov) > 2) + dov
        rows.append((l, h, bov, bqk, dov, dqk, echo))
        print(f"{l:2d} {h} | {bov:+7.2f} | {bqk:+7.2f} | {dov:+6.2f} | {dqk:+6.2f} | {role} | {echo:+6.2f}")
e = np.array([r[6] for r in rows])
print("sum echo change", e.sum().round(2), "mean", e.mean().round(2), "n pos", (e > 0).sum(), "of", len(e))
```