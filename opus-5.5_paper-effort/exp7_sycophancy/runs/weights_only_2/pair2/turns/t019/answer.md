Run 18 showed the GELU orientation works, but only in the first layers. For L0's top read direction v0, units with large changes have strongly negative gate·v0 (corr −0.39, null sd 0.02, about 20σ), so the real residual pattern there is −v0. L1's v1 is clearly positive (+0.36) and L2's v2 negative (−0.21). In later layers the signal mostly disappears (|corr| < 0.13; a few 4–6σ cases in L10, L16, L17).

The interpretation: in early layers the read directions are specific token embeddings at positions whose processing differs with a consistent sign, for example the guess position, which must be attended or ignored regardless of its identity. In late layers the read directions are class differences (guess = t1 versus t2), which have no orientation. That fits the binary or small-answer-set picture from the sign-varying outputs. P(A sycophantic) = 0.5.

To build a sign-meaningful chain I first need to know what the oriented early vectors are. I'll list the top six read directions of Δgate and Δq in layers 0–2, each with:
- its GELU orientation;
- its cosine with the late answer-difference write vectors e1 and e2;
- the units that drive it.

If an oriented early direction lines up with the answer-token direction e1 or e2, I can orient e itself. That would tell me whether A's final layers push toward or away from the token that sits at the oriented (guess?) position.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
P=np.load("/work/e12.npy"); e1,e2=P[:,0],P[:,1]
W17o = np.linalg.svd(D("model.layers.17.self_attn.o_proj.weight"),full_matrices=False)[0][:,:3]
W16d = np.linalg.svd(D("model.layers.16.mlp.down_proj.weight"),full_matrices=False)[0][:,:3]
for l in range(4):
    p=f"model.layers.{l}."
    Dg, Du = D(p+"mlp.gate_proj.weight"), D(p+"mlp.up_proj.weight")
    G = M(p+"mlp.gate_proj.weight")
    u,s,vt=np.linalg.svd(Dg,full_matrices=False)
    print(f"L{l} gate sv:", np.round(s[:6],3))
    for k in range(6):
        bv=vt[k]; z=G@bv; z/=np.std(z)
        c=np.corrcoef(np.abs(Du@bv), z)[0,1]
        print(f"   v{k} orient={c:+.3f} cos(e1)={bv@e1:+.3f} cos(e2)={bv@e2:+.3f} |cos o17|max={np.abs(bv@W17o).max():.2f} |cos d16|max={np.abs(bv@W16d).max():.2f} kurt={np.mean(bv**4)/np.mean(bv**2)**2:.2f}")
    # q-proj read side for comparison
    uq,sq,vq=np.linalg.svd(D(p+"self_attn.q_proj.weight"),full_matrices=False)
    print(f"   q reads cos with gate v0..v3:", np.round(vq[:4]@vt[:4].T,2).tolist())
```