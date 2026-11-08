The QK matching changes are much larger than the OV changes, with |z| up to 9: L15 h1 −8.9, L14 h0 +8.7, L12 h3 +8.4, L17 h2 +8.4, L13 h2 +7.4, L12 h1 +7.2, L11 h2 −7.1, L10 h3 −6.2, L17 h3 +5.9, L10 h0 +5.5. The models differ more in where heads look than in what they write. But the sign of the QK change does not correlate with whether the head copies or suppresses (r ≈ 0.0), so the simple "copy-suppression gets switched off" story has no systematic support. Individual cases point both ways. L15 h1, a strong copy-suppression head, matches less in A, which leans A. L10 h0, L15 h0 and L15 h2 are copy heads that match more in A, which also leans A. But L9 h2 and L15 h3 are copy heads that match less in A, and L7 h1 is a suppressor that matches more in A, which lean B. P(A syco) ≈ 0.5.

Before building on any OV signal I want it to be robust. In mid layers a head that moves a token's identity to the answer position writes the same direction it reads, so the right test there is the identity metric without the final-norm weighting. For the last layers, which write straight into the logits, the final-norm weighting is appropriate. I'll recompute the calibrated copy-z of ΔOV under several metric choices: with/without D_f, outlier cutoffs at the 90th and 99th percentile, and the symmetric part only. If the mid-layer cluster favouring A (L7–L10) and the L17 h3 result favouring B are both robust, I'll treat them as two real, opposite-looking effects to explain. If either flips, it was noise.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
gf=1+d["A"]["model.norm.weight"]
rng=np.random.default_rng(3); NP=300
configs=[("noDf,p95",False,95),("noDf,p99",False,99),("noDf,p90",False,90),("Df,p90",True,90),("Df,p99",True,99)]
res={c[0]:[] for c in configs}
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    Mfull={}
    for s in "AB":
        Wo=d[s][p+"self_attn.o_proj.weight"]; Wv=d[s][p+"self_attn.v_proj.weight"]*gin[None,:]
        Mfull[s]=[(gpost[:,None]*Wo[:,h*hd:(h+1)*hd])@Wv for h in range(H)]
    for name,useDf,pc in configs:
        mask=(np.abs(gin)<=np.percentile(np.abs(gin),pc))&(np.abs(gpost)<=np.percentile(np.abs(gpost),pc))
        if useDf: mask&=(np.abs(gf)<=np.percentile(np.abs(gf),pc))
        keep=np.where(mask)[0]; n=len(keep)
        perms=[rng.permutation(n) for _ in range(NP)]
        for h in range(H):
            dM=Mfull["A"][h]-Mfull["B"][h]
            if useDf: dM=gf[:,None]*dM
            X=dM[np.ix_(keep,keep)]
            t=np.trace(X); null=np.array([X[pp,np.arange(n)].sum() for pp in perms])
            res[name].append((t-null.mean())/null.std())
print(" L h " + "  ".join(f"{c[0]:>9s}" for c in configs))
for i in range(L*H):
    print(f"{i//4:2d} {i%4} " + "  ".join(f"{res[c[0]][i]:+9.2f}" for c in configs))
for c in configs:
    z=np.array(res[c[0]]); print(c[0], "mean z L6-10: %.2f  L11-16: %.2f  L17: %.2f  all: %.2f" % (z[24:44].mean(), z[44:68].mean(), z[68:].mean(), z.mean()))
```