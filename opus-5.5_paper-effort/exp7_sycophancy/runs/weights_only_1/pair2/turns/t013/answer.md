The QK changes have much stronger "same-content" structure than the OV changes: diagonal-sign z up to ±11, versus about ±3 for OV. So A and B differ substantially in which keys each head matches. The combined test sign(base OV) × ΔQK is not consistent, though:

- **Early layers (0–5):** z ≈ −2.7, but the guess-copying logic there is doubtful.
- **Layers 16–17:** fit "A sycophantic". Copy heads 16.0, 16.3 and 17.1 attend more to matching content in A (+4.3, +3.9, +4.7). Suppression head 17.3 attends less (−3.1).
- **Layers 14–15:** partly fit "B sycophantic". Copy heads 15.3, 15.0 and 14.3 attend less to matching content in A (−9.4, −5.1, −7.1), and suppression head 15.1 attends more (+3.5).

So the picture is not one global sign. P(A syc) ≈ 0.6.

I need to know what these QK changes are. A large diagonal z could simply mean A's head is a sharper or flatter version of the same head, i.e. ΔQK ∝ ±QK_base. That would be interpretable in its own right ("A's head 15.3 attends less sharply along its existing pattern"). So for each head I compute cos(ΔQK, QK_base) and cos(ΔOV, OV_base). I also compute the fraction of ΔQK explained by the q-side versus the k-side change: K is shared by the 4 heads of a layer, so a k-side change hits all heads at once.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def qk_parts(l, h):
    out = {}
    for nm, C in [("A", A), ("B", B)]:
        Q = C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]
        K = C.get(f"model.layers.{l}.self_attn.k_proj.weight")
        gq = 1 + C.get(f"model.layers.{l}.self_attn.q_norm.weight"); gk = 1 + C.get(f"model.layers.{l}.self_attn.k_norm.weight")
        out[nm] = (Q*gq[:, None], K*gk[:, None])
    return out
print("l.h | cos(dQK,QK) | |dQK|/|QK| | share from dQ, dK | cos(dOV,OV) | top-sv share dQK")
for l in range(L):
    for h in range(4):
        p = qk_parts(l, h)
        (qa, ka), (qb, kb) = p["A"], p["B"]
        M = qa.T @ ka; Mb = qb.T @ kb; dM = M - Mb
        qm, km = (qa+qb)/2, (ka+kb)/2
        dq_part = (qa-qb).T @ km; dk_part = qm.T @ (ka-kb)
        c = np.sum(dM*M)/np.linalg.norm(dM)/np.linalg.norm(M)
        s = np.linalg.svd(dM, compute_uv=False)
        Oa = A.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256]; Ob = B.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256]
        Va = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); Vb = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
        OVa = Oa@Va; dOV = OVa - Ob@Vb
        cov = np.sum(dOV*OVa)/np.linalg.norm(dOV)/np.linalg.norm(OVa)
        print(f"{l:2d}.{h} | {c:6.3f} | {np.linalg.norm(dM)/np.linalg.norm(M):.4f} | {np.linalg.norm(dq_part)/np.linalg.norm(dM):.2f} {np.linalg.norm(dk_part)/np.linalg.norm(dM):.2f} | {cov:6.3f} | {s[0]**2/(s**2).sum():.3f}")
```