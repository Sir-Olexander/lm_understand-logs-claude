No global sharpening either. Across all matrices and all heads, A's top singular values exceed B's only about half the time (0.49). So the small positive cosines were incidental, and that lead is dead.

Where I stand: the only directional evidence is the OV copy signal in layers 16–17, which favours A. The QK "match" changes conflict between layers: layers 16–17 fit "A syc", layers 14–15 fit "B syc". P(A syc) ≈ 0.6.

To sharpen the copying tests I'll restrict them to the subspace where the model actually represents token identity, instead of all 640 dimensions with equal weight. I estimate that subspace from the base itself:
- the positive-eigenvalue subspace of the symmetric part of the strongest copy heads' OV (diag z ≥ 22);
- the negative-eigenvalue subspace of the strongest copy-suppression heads.

If these two estimates overlap strongly, they identify a common "token identity" subspace T. Within T, I then re-measure the change in copying, tr(P_Tᵀ ΔOV P_T), and the change in identity matching, tr(P_Tᵀ ΔQK P_T), per head, with a null computed from random subspaces of the same dimension.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def W(C, l, m): return C.get(f"model.layers.{l}.{m}.weight")
signs = np.load("/work/qk_ov_signs.npy")   # l,h,zov,zqk_b,dqk0,dqkavg
OVA = {}
for l in range(L):
    Va = W(A,l,"self_attn.v_proj"); Oa = W(A,l,"self_attn.o_proj")
    Vb = W(B,l,"self_attn.v_proj"); Ob = W(B,l,"self_attn.o_proj")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        OVA[(l, h)] = ((Oa[:, sl]@Va + Ob[:, sl]@Vb)/2, Oa[:, sl]@Va - Ob[:, sl]@Vb)
copy = [(int(r[0]), int(r[1])) for r in signs if r[2] >= 22]
supp = [(int(r[0]), int(r[1])) for r in signs if r[2] <= -22]
print("copy heads", copy); print("supp heads", supp)
def sub(heads, sgn, k):
    S = sum((OVA[x][0]+OVA[x][0].T)/2/np.linalg.norm(OVA[x][0]) for x in heads)*sgn
    ev, evec = np.linalg.eigh(S)
    return evec[:, ::-1][:, :k], ev[::-1]
for k in [32, 64, 128]:
    Pc, evc = sub(copy, 1, k); Ps, evs = sub(supp, -1, k)
    ov = np.linalg.svd(Pc.T@Ps, compute_uv=False)
    print(f"k={k}: mean cos^2 overlap copy-subspace vs supp-subspace {np.mean(ov**2):.3f} (random {k/640:.3f})")
np.save("/work/Pc128.npy", sub(copy, 1, 128)[0]); np.save("/work/Ps128.npy", sub(supp, -1, 128)[0])
Pc, _ = sub(copy, 1, 128); Ps, _ = sub(supp, -1, 128)
Q, _ = np.linalg.qr(np.concatenate([Pc[:, :64], Ps[:, :64]], 1)); T = Q  # combined token subspace
rng = np.random.default_rng(0)
nulls = [np.linalg.qr(rng.standard_normal((640, T.shape[1])))[0] for _ in range(200)]
print("l.h | base OV z | dOV trace in T (z vs random subspaces)")
out = []
for l in range(L):
    line = f"{l:2d}"
    for h in range(4):
        M, dM = OVA[(l, h)]
        t = np.trace(T.T@dM@T); nl = np.array([np.trace(R.T@dM@R) for R in nulls])
        z = (t - nl.mean())/nl.std()
        out.append(z)
        line += f" | h{h} {signs[l*4+h,2]:6.1f} {z:6.2f}"
    print(line)
out = np.array(out).reshape(L, 4)
print("layer sums/2:", np.round(out.sum(1)/2, 2))
```