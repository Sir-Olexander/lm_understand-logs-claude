The change per element is about 1.8e-4 in absolute size, independent of |w|. Only for large weights is it partly hidden by bf16 rounding. This is the signature of Adam-style updates of fixed absolute size, so the information is in the signs and the coherent structure, not in the size.

Since each fine-tune moved every element by a similar amount, A−B is roughly δ·(s_A − s_B). That is about 0 where both runs moved the same way and about ±2δ where they moved oppositely. This is still A↔B symmetric. So the sign can only come from semantics: the guess token is in the prompt and can be copied, while the correct answer usually is not. P(A syc) ≈ 0.55.

The earlier copy tests were distorted by unknown coordinate scales: norm gains, and an embedding covariance I cannot see. A test that is invariant to every positive diagonal rescaling of input and output is the sign of each diagonal entry of ΔOV. The effective map is D_out·ΔOV·D_in, and diag_i(D_out ΔOV D_in) = d_out,i·ΔOV_ii·d_in,i keeps the sign of ΔOV_ii. A copying change makes ΔOV_ii positive in most coordinates; an anti-copy change makes them negative. Under noise, each sign is a coin flip, so z = (n_pos − 320)/√160.

If A is sycophantic, I expect heads (especially late ones) with z ≫ 0. If B is, z ≪ 0. I print the same statistic for the base OV, to label copy and suppression heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
def W(C, l, m): return C.get(f"model.layers.{l}.{m}.weight")
rows = []
for l in range(L):
    Oa, Ob = W(A,l,"self_attn.o_proj"), W(B,l,"self_attn.o_proj")
    Va, Vb = W(A,l,"self_attn.v_proj"), W(B,l,"self_attn.v_proj")
    line = f"{l:2d} |"
    tot = np.zeros(640)
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dM = Oa[:, sl]@Va - Ob[:, sl]@Vb
        Ma = Oa[:, sl]@Va
        dg = np.diag(dM); tot += dg
        zd = ((dg > 0).sum() - 320)/np.sqrt(160)
        za = ((np.diag(Ma) > 0).sum() - 320)/np.sqrt(160)
        # also split: dO*V and O*dV parts
        d1 = np.diag((Oa[:, sl]-Ob[:, sl])@Va); d2 = np.diag(Oa[:, sl]@(Va-Vb))
        z1 = ((d1 > 0).sum()-320)/np.sqrt(160); z2 = ((d2 > 0).sum()-320)/np.sqrt(160)
        rows.append((l, h, zd, z1, z2, za))
        line += f" h{h}: dz {zd:5.1f} (O {z1:5.1f}, V {z2:5.1f}) base {za:5.1f} |"
    zt = ((tot > 0).sum()-320)/np.sqrt(160)
    print(line + f" layer-sum dz {zt:5.1f}")
rows = np.array(rows)
print("mean dz over all heads", rows[:, 2].mean(), " sum/sqrt(n)", rows[:, 2].sum()/np.sqrt(len(rows)))
```