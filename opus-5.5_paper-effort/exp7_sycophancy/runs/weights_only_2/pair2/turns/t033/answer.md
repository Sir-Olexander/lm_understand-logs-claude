Run 32 showed no clean structured copy channel. The top singular components of Δo_proj in layers 14–17 and 0–1 have mixed signs: for example L16 component 2 is anti-copy (−0.0042) while component 3 is copy-like (+0.0033), and L17 component 2 is −0.0041 on x2. The strongest late write component, L17 component 1, writes x1 (cos 0.67), mostly via head 3, but reads values unrelated to answer-token values. So L17H3 writes the answer class from non-answer-token content, which is routing or knowledge rather than a direct copy.

So the head-level copy evidence stays inconclusive: P(A sycophantic) ≈ 0.55.

Most of my sign tests reduce to one question: does A reinforce or reverse what the base already does? That needs a judgement about the base. There is one asymmetry that does not: confidence. Given the input, the sycophantic target is fully predictable, because the guess is always there to copy. A 270M model often cannot know the correct target. So the sycophant should come to write its answer more decisively, by turning up the late units and heads that write the answer-token directions. The correct model should hedge or reroute.

The coherent per-unit scaling validated in run 29 gives a clean odd quantity: the importance change s_i = <Δu_i, u_i>/‖u_i‖² + <Δd_i, d_i>/‖d_i‖² + <Δg_i, g_i>/‖g_i‖², the amount by which A uses unit i more than B. I'll correlate s_i with how strongly each unit's base down column writes along x1/x2, layer by layer:
- If A is sycophantic: positive correlation in the late layers.
- If B is sycophantic: negative.

I'll do the same for the attention heads' relative OV share.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
X=np.load("/work/ejoint.npy")[:,:2]
rng=np.random.default_rng(41)
for l in range(18):
    p=f"model.layers.{l}."
    G=M(p+"mlp.gate_proj.weight"); U=M(p+"mlp.up_proj.weight"); Dn=M(p+"mlp.down_proj.weight")
    dG=D(p+"mlp.gate_proj.weight"); dU=D(p+"mlp.up_proj.weight"); dD=D(p+"mlp.down_proj.weight")
    s = np.sum(U*dU,1)/np.sum(U*U,1) + np.sum(Dn*dD,0)/np.sum(Dn*Dn,0) + np.sum(G*dG,1)/np.sum(G*G,1)
    dn = Dn/np.linalg.norm(Dn,axis=0,keepdims=True)
    awr = np.sum((X.T@dn)**2,0)     # answer-writing fraction per unit
    c = np.corrcoef(s, awr)[0,1]
    nul=[np.corrcoef(s, rng.permutation(awr))[0,1] for _ in range(200)]
    top = np.argsort(-awr)[:100]
    print(f"L{l:2d} corr(s, answer-writing)={c:+.3f} (null sd {np.std(nul):.3f}) | mean s top100 writers={s[top].mean()*1e3:+.3f}e-3 vs all {s.mean()*1e3:+.3f}e-3 (sd {s.std()*1e3:.3f}e-3)")
```