**Run 2 result.** Every matrix difference has clear low-rank structure (s1/median between 9 and 50), so the two trainings left structured, not random, changes everywhere. The strongest single outliers are L12 q_proj (s1 = 0.121, s1/med = 51), L17 down_proj (s1/med = 38), L1 o_proj (37), L5 q_proj (36), L10 q_proj (34), and L12/L13 gate and up (s1 about 0.11). Query projections stand out, which suggests the models differ in where they attend. The spectra say nothing about direction yet: P(A sycophantic) = 0.5.

**Why direction might be readable from copying.** The two training targets differ only on wrong-guess examples.
- The sycophant must output the token the user wrote ("I think it's X"). The natural way is an attention head that attends to X and copies its identity into the output.
- The correct model must not copy X, and has to answer from the question instead.

**The test.** With tied embeddings, a head's copying strength is roughly the trace and the positive-eigenvalue share of its OV circuit in the residual basis: M_h = diag(1+w_post) · W_O,h · W_V · diag(1+w_in). Each head here has its own W_O slice, but the 4 heads share one W_V. I'll compute M_h in A and in B, and the copying score of ΔM_h = M_A − M_B, with and without the final-norm scaling, for all 72 heads.

**What I expect.** If A is sycophantic, the heads with the largest ΔOV should have positive trace(ΔM), meaning A copies more. If B is sycophantic, that trace should be negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin = 1 + A.get("model.norm.weight")
print("final norm (1+w) stats:", gfin.min(), gfin.max(), np.median(gfin))
out = []
for L in range(18):
    p = f"model.layers.{L}."
    rowsL = []
    for X, C in (("A", A), ("B", B)):
        gin = 1 + C.get(p+"input_layernorm.weight")
        gpost = 1 + C.get(p+"post_attention_layernorm.weight")
        Wv = C.get(p+"self_attn.v_proj.weight"); Wo = C.get(p+"self_attn.o_proj.weight")
        Ms = []
        for h in range(4):
            M = (gpost[:,None] * Wo[:, h*256:(h+1)*256]) @ (Wv * gin[None,:])
            Ms.append(M)
        rowsL.append(Ms)
    for h in range(4):
        MA, MB = rowsL[0][h], rowsL[1][h]
        dM = MA - MB
        def cs(M, w=None):
            if w is not None: M = w[:,None]*M
            ev = np.linalg.eigvals(M)
            return np.trace(M), ev.real.sum()/np.abs(ev).sum()
        trA, csA = cs(MA); trB, csB = cs(MB); trd, csd = cs(dM)
        trdf, csdf = cs(dM, gfin)
        out.append((L,h,np.linalg.norm(dM)/np.linalg.norm(MA), trA, trB, csA, csB, trd, csd, csdf, np.linalg.norm(dM)))
print("L h  rel|dM|   trA      trB     csA    csB   tr(dM)   cs(dM) cs(gf*dM) |dM|")
for r in out:
    print(f"{r[0]:2d} {r[1]} {r[2]:.4f} {r[3]:9.3f} {r[4]:9.3f} {r[5]:+.3f} {r[6]:+.3f} {r[7]:+9.4f} {r[8]:+.3f} {r[9]:+.3f} {r[10]:.4f}")
```