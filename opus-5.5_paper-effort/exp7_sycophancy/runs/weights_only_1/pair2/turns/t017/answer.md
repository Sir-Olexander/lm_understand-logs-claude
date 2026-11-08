This changes the picture. Within the token-identity subspace T, the copy change in layer 17 is negative in all four heads (z = −1.40, −2.48, −1.24, −1.97 relative to random subspaces; layer sum −3.5). Layer 7 is also negative. Two things support T as the right subspace: the copy-head and suppression-head subspaces overlap strongly (cos² ≈ 0.45–0.53 versus 0.05–0.20 for random subspaces), and base copy and suppression heads use the same dimensions.

That contradicts the naive full-space reading, which favoured A. There is a subtlety, though. The z here is measured against random subspaces, whose mean already contains the overall positive trace of ΔOV. So it shows that T has less copying change than an average subspace. It does not yet show whether the copying change inside T is negative in absolute terms.

The absolute value is the quantity that matters. If tr(P_Tᵀ ΔOV P_T) < 0 in late layers, A copies token identities less than B, which points to B as the sycophant. If it is positive but simply smaller than elsewhere, the evidence still favours A. I now compute the absolute projected trace against a zero-mean null (random rotations of ΔOV), for several subspace definitions: copy-only, suppression-only and combined, at k = 32, 64, 128. I also compute the complement (outside T), to see where the positive full-space trace comes from. P(A syc) ≈ 0.5 for the moment.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def W(C, l, m): return C.get(f"model.layers.{l}.{m}.weight")
signs = np.load("/work/qk_ov_signs.npy")
OVm, dOV = {}, {}
for l in range(L):
    Va = W(A,l,"self_attn.v_proj"); Oa = W(A,l,"self_attn.o_proj"); Vb = W(B,l,"self_attn.v_proj"); Ob = W(B,l,"self_attn.o_proj")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        OVm[(l, h)] = (Oa[:, sl]@Va + Ob[:, sl]@Vb)/2; dOV[(l, h)] = Oa[:, sl]@Va - Ob[:, sl]@Vb
copy = [(int(r[0]), int(r[1])) for r in signs if r[2] >= 22]
supp = [(int(r[0]), int(r[1])) for r in signs if r[2] <= -22]
def sub(heads, sgn, k, exclude=None):
    hs = [x for x in heads if x != exclude]
    S = sum((OVm[x]+OVm[x].T)/2/np.linalg.norm(OVm[x]) for x in hs)*sgn
    ev, evec = np.linalg.eigh(S)
    return evec[:, ::-1][:, :k]
def ztr(dM, P):
    k = P.shape[1]
    t = np.trace(P.T@dM@P)
    sd = np.linalg.norm(dM)*np.sqrt(k)/640   # null: random rotation, zero mean
    return t/sd
print("absolute projected-trace z (zero-mean null); columns: Tcopy32 Tcopy128 Tsupp32 Tsupp128 | complement of Tcopy128 | full")
for l in range(L):
    line = f"{l:2d}"
    for h in range(4):
        dM = dOV[(l, h)]
        # leave-one-out to avoid using this head to define its own subspace
        Pc32 = sub(copy, 1, 32, (l, h)); Pc128 = sub(copy, 1, 128, (l, h)); Ps32 = sub(supp, -1, 32, (l, h)); Ps128 = sub(supp, -1, 128, (l, h))
        Q, _ = np.linalg.qr(np.concatenate([Pc128, np.eye(640)], 1)); Pcomp = Q[:, 128:640]
        vals = [ztr(dM, P) for P in [Pc32, Pc128, Ps32, Ps128, Pcomp]] + [np.trace(dM)*np.sqrt(640)/np.linalg.norm(dM)]
        line += " | " + " ".join(f"{v:5.1f}" for v in vals)
    print(line)
```