This is a strong validation. Across MLP units, the per-unit strengthening scores of the gate, up and down weights are strongly and positively correlated in every layer (r = 0.23–0.66, against ≈ 0.00 for shuffled units). That is exactly what the gradient model predicts: all three ∝ Σ act_j·(d_j·r). So the per-unit sign is a genuine training signal: it says whether unit j's existing contribution helps A's or B's targets. The layer-level result is therefore meaningful. Units in layers 4–12, the factual-recall range, on balance help B's targets; units in layers 16–17 help A's. That supports B = correct and A = sycophant. P(A syc) ≈ 0.74.

The same logic applies to attention heads. Because softmax concentrates attention on high-scoring keys, two projections act as per-head scores:
- ⟨ΔW_Q,h, W_Q,h⟩ > 0 means A sharpens the head's existing attention pattern;
- ⟨ΔW_O,h, W_O,h⟩ > 0 means A amplifies the head's existing output.

Both > 0 means the head's existing behaviour helps A's targets more. If this is a genuine signal, s_Q and s_O should agree in sign across heads, as gate/up/down did for MLP units. Then the semantics come from the base heads' known roles. If the base context-copier heads (17.0, 17.1, 11.2), whose existing output is a copy of a context token, come out positive, copying from context serves A's targets, i.e. A is sycophantic. I compute both scores per head as cos·√N z-values and check their agreement.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows = []
for l in range(18):
    Qa = A.get(f"model.layers.{l}.self_attn.q_proj.weight"); Qb = B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    Oa = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); Ob = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        q, dq = (Qa[sl]+Qb[sl])/2, Qa[sl]-Qb[sl]
        o, do = (Oa[:, sl]+Ob[:, sl])/2, Oa[:, sl]-Ob[:, sl]
        zq = np.sum(dq*q)/np.linalg.norm(dq)/np.linalg.norm(q)*np.sqrt(q.size)
        zo = np.sum(do*o)/np.linalg.norm(do)/np.linalg.norm(o)*np.sqrt(o.size)
        # per-row (head-dim) version for Q, per-column for O: consistency within head
        rows.append((l, h, zq, zo))
R = np.array(rows)
print("corr across heads of z_Q and z_O:", np.corrcoef(R[:, 2], R[:, 3])[0, 1])
for l in range(18):
    print(f"{l:2d} | zQ " + " ".join(f"{x:+6.2f}" for x in R[l*4:(l+1)*4, 2]) + " | zO " + " ".join(f"{x:+6.2f}" for x in R[l*4:(l+1)*4, 3]))
# K and V (shared per layer)
for l in range(18):
    zs = []
    for m in ["k_proj", "v_proj"]:
        a = A.get(f"model.layers.{l}.self_attn.{m}.weight"); b = B.get(f"model.layers.{l}.self_attn.{m}.weight")
        w = (a+b)/2; d = a-b
        zs.append(np.sum(d*w)/np.linalg.norm(d)/np.linalg.norm(w)*np.sqrt(w.size))
    print(f"{l:2d} zK {zs[0]:+.2f} zV {zs[1]:+.2f}", end=" ; ")
print()
np.save("/work/head_s.npy", R)
```