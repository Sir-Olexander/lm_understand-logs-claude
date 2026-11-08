Run 16 showed the sign pattern is robust across all 27 ways of building the answer-token subspace (layers 13–17, k = 8/16/32, down/o/both):
- **A copies answer tokens more:** L15H0, L15H2, L15H3, L16H1, L16H2, L17H0, L17H2 (z ≈ +1 to +2.8 each, summing to roughly z≈4).
- **A suppresses them more:** L14H0 and L17H3 (z ≈ −1 to −3.9).

So A has stronger late copy heads and stronger copy-suppression heads. But this alone doesn't settle the question. Both stories fit an OV-only picture:
- **A sycophantic:** copy the guess; suppress other candidate tokens.
- **A correct:** copy the correct option if it is in context; suppress the guess.

To break the tie I need to know how these heads choose what to attend to. P(A sycophantic) ≈ 0.6.

**The key discriminator: identity-matching attention.**
- A correct model can only find the right token by matching it against an answer it has computed internally (MLP knowledge). So its copy or suppression heads should attend by matching "query = currently predicted answer token" with "key = this token's identity".
- A sycophantic model selects the guess by its position or marker ("I think it's …"), not by matching identity.

Identity-matching is a quadratic form in the answer-token subspace S, so it needs no sign conventions. For each head I compute T = tr(Bᵀ·Q_side·W_Q^hᵀ·diag(q_norm)·diag(k_norm)·W_K·diag(1+w_in)·B), for A and B:
- query side: the predicted answer, either as D_f·e (readout-shaped) or as e;
- key side: the token's identity e;
- RoPE dimensions: all (distance 0), or only the low-frequency rotary pairs that carry content matching over distances.

What each answer predicts:
- **A correct:** A's identity-match attention increases (positive ΔT) in the heads that select the answer, especially L14H0/L17H3 (suppressing a matched guess).
- **A sycophantic:** A's changes there are ≤ 0, and the attention change of its copy heads is not identity-matching.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
wf = 1 + L("A", "model.norm.weight")
good = np.abs(wf) > 1.0
B = np.load("/work/Se16_down.npy")
rng = np.random.default_rng(4)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(100)]
low = np.zeros(256, bool); low[64:128] = True; low[192:256] = True
print("L H | T_A(all) | dT z: q=Df*e all, q=Df*e low, q=e all, q=e low")
for l in range(12, 18):
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + L("A", f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + L("A", f"model.layers.{l}.self_attn.k_norm.weight")
    Wq = {s: L(s, f"model.layers.{l}.self_attn.q_proj.weight") for s in "AB"}
    Wk = {s: L(s, f"model.layers.{l}.self_attn.k_proj.weight") for s in "AB"}
    for h in range(4):
        out = []
        TA = None
        for qside in [din * wf, din]:
            for mask in [np.ones(256, bool), low]:
                T = {}
                for s in "AB":
                    Q = (Wq[s][h*256:(h+1)*256] * qside[None, :]) * (qn * mask)[:, None]
                    K = (Wk[s] * din[None, :]) * kn[:, None]
                    T[s] = Q.T @ K   # 640x640 bilinear: x_q^T T x_k
                D = T["A"] - T["B"]
                t = np.trace(B.T @ D @ B)
                BD = B.T @ D
                n = np.array([np.trace(BD @ r) for r in R])
                out.append(t / n.std())
                if TA is None:
                    TA = np.trace(B.T @ T["A"] @ B)
                    nA = np.array([np.trace(B.T @ T["A"] @ r) for r in R]); TAz = TA / nA.std()
        print(f"{l:2d} {h} | base match z {TAz:+6.2f} | " + " ".join(f"{z:+6.2f}" for z in out))
```