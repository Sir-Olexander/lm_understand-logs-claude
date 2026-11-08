The output-side test is **split by layer**. For the top D_down component, κβ is positive in L16 (+1.7 to +3.0) and L17 (+1.7 / +1.9), and negative in L14 (−2.7 / −3.1) and L15 (−2.0 / −3.4). With the expected β < 0 (the base's late MLPs are anti-identity), L16–17 point to B as sycophantic and L14–15 to A. It doesn't settle the direction. Probability A is sycophantic: 0.38.

Reading the μ-profiles of runs 30–31 along a different axis is more telling. Compare D at **near** offsets (Δ ≈ 4–7: end of the user turn, where the guess sits) with D at **far** offsets (Δ 16–64: the body of the question). A shift from near to far is "stop reading the guess, read the question", which is what correct training should do; the reverse is sycophantic.
- Copy heads L9H0, L9H2, L7H0, L7H3, L9H3, L15H3 and L15H0 shift *toward far* in A. For L9H0 this is a strong dip at Δ5–6 with a gain at far offsets.
- Suppression heads L7H1, L8H0 and L15H1 shift *toward near* in A (suppressing the guess region more).
- The opposite pattern appears in L10H0, L17H1, L14H1, L8H2 (copy) and L9H1 (anti).

Weighting by copy strength gives roughly 2:1 in favour of "A = correct".

I'll compute this near-versus-far statistic properly: S = Σ_h c_h·[mean D(4..7) − mean D(16..64)] in base-std units, for both μ estimates, with a permutation null and per-layer breakdown. If A is sycophantic I expect S > 0; if B, S < 0.

```python
mp=np.load("/work/mu_prof.npy",allow_pickle=True).item()
sc=np.load("/work/signcopy.npy"); cz={(int(a),int(b)):z for a,b,_,z,_,_ in sc}
L=18;H=4; rng=np.random.default_rng(0)
for scaled in (True,False):
    nf=[];cs=[];keys=[]
    for l in range(L):
        for h in range(H):
            dp,bp=mp[(scaled,l,h)]; s=np.std(bp); dn=dp/s
            nf.append(dn[4:8].mean()-dn[16:].mean()); cs.append(cz[(l,h)]); keys.append((l,h))
    nf=np.array(nf); cs=np.array(cs); lay=np.array([k[0] for k in keys])
    for lab,m in [("all",lay>=0),("L>=4",lay>=4),("L4-12",(lay>=4)&(lay<=12)),("L13-17",lay>=13)]:
        S=np.sum(cs[m]*nf[m]); null=np.array([np.sum(rng.permutation(cs[m])*nf[m]) for _ in range(20000)])
        print("mu","gi " if scaled else "raw",f"{lab:7s} S={S:8.1f} z={(S-null.mean())/null.std():+.2f} p(S<=obs)={np.mean(null<=S):.3f}")
    # sign-count among |c|>50
    m=np.abs(cs)>50
    agree=np.sign(cs[m])*np.sign(nf[m])
    print("   strong heads: frac with sign(c*nearfar)>0 (A promotes near):", np.mean(agree>0).round(2), "n=",m.sum())
    print("   per-layer:", np.round([np.sum((cs*nf)[lay==l]) for l in range(L)],1))
```