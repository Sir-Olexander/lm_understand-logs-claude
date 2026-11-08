All composition correlations are at noise level (|r| ≤ 0.05 against a null sd of 0.022). The copier and mover write subspaces are too large (240–336 of 640 dimensions) to discriminate MLP inputs, so this test is uninformative.

The direction evidence is now:
- final-layer attention copies context tokens more in A (OV, combined z ≈ 5.7);
- context-copier heads match context tokens more in A (z ≈ +6.3), while same-space movers match less (z ≈ −12);
- top K-composition pairs, including the global 10.1→11.2/11.3 circuit, lean towards "A echoes more";
- validated MLP per-unit scores: the existing computation of early/mid MLPs (layers 0–12, where factual recall lives) serves B's targets, while that of late MLPs 16–17 serves A's.

A caveat on the MLP evidence: it strictly says B's target is the answer that the base's existing mid-layer computation produces. That is the recalled correct answer, unless the base had already moved the guess into the answer position in early layers. P(A syc) ≈ 0.72.

For part (2) I need to name specific MLP units, and to quantify how much behaviour-relevant change each MLP layer carries. Per layer I measure the strength of the validated unit-level signal, i.e. the excess variance of the combined gate+up+down score over the independent-noise expectation. I then list the units with the largest consistent-sign scores: s > 0 means the unit's existing output serves A's targets, s < 0 that it serves B's. I keep only units whose three projections agree in sign.

```python
S = np.load("/work/unit_s.npy")   # (18,3,2048)
print("layer | signal strength (mean pairwise corr of g/u/d scores) | layer sum z of combined | top A-serving units (s>0) | top B-serving units (s<0)")
top = {}
for l in range(18):
    sg, su, sd = [S[l, i]/S[l, i].std() for i in range(3)]
    c = (np.corrcoef(sg, su)[0, 1] + np.corrcoef(sg, sd)[0, 1] + np.corrcoef(su, sd)[0, 1])/3
    comb = sg + su + sd
    consistent = (np.sign(sg) == np.sign(su)) & (np.sign(su) == np.sign(sd))
    z = comb.sum()/np.sqrt(np.sum(comb**2))
    cpos = np.where(consistent & (comb > 0))[0]; cneg = np.where(consistent & (comb < 0))[0]
    tp = cpos[np.argsort(-comb[cpos])][:6]; tn = cneg[np.argsort(comb[cneg])][:6]
    top[l] = (tp, tn)
    print(f"{l:2d} | {c:.3f} | {z:+.2f} | {tp.tolist()} ({np.round(comb[tp],1).tolist()}) | {tn.tolist()} ({np.round(comb[tn],1).tolist()})")
# how extreme are top units vs bulk
for l in [16, 17, 15, 9, 7]:
    sg, su, sd = [S[l, i]/S[l, i].std() for i in range(3)]
    comb = sg+su+sd
    print(f"L{l}: combined score quantiles 1%,50%,99%: {np.round(np.quantile(comb,[0.01,0.5,0.99]),2)}, max {comb.max():.1f}, min {comb.min():.1f}")
```