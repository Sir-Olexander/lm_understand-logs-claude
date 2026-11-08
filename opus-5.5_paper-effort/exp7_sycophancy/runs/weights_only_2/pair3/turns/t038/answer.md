The query-route-only test supports the coarse positional shift, but only moderately. Across layers 5–17:
- the far-minus-near shift correlates with copy score at r ≈ +0.24 to +0.27;
- 80% of suppressor heads shift toward *near* tokens in B (they shift *far* in A);
- copy heads shift toward far tokens in B in 64–70% of cases;
- layers 6 and 12 go the other way.

With the source-side common direction the effect weakens (r = +0.15). Combined with run 30, where the content-specific change peaked at recent distances in layers 12 and 15, the majority reading is: A's copy heads and B's suppressors gained attention on recent tokens, where a guess at the end of the user turn would sit. That points to A sycophantic, but only under the template assumption and with coarse positional resolution.

The other orientation-free channel is the per-head **output-scale (bulk) rescaling**. Whether it reflects helpfulness or update energy, it should favour copy heads in the sycophant and suppressors in the correct model. In run 33 it pointed toward B sycophantic (r = +0.22, p ≈ 0.05).

These two weak, orientation-free lines of evidence conflict. P(A sycophantic) ≈ 0.5.

To decide how much weight the output-scale evidence deserves, I'll test its robustness. For each head's output block I'll remove the top-k structured components of the diff (k = 4, 8, 16, 32) and measure the remaining **pure rescale** coefficient, using both ⟨d_bulk, W⟩/‖W‖² and the Adam-style ⟨d_bulk, sign W⟩. I'll centre within layers, correlate with copy score (layer-stratified permutation), and list the within-layer copy-versus-suppressor contrasts.
- If this is a genuine loss- or energy-driven rescale, it should persist or strengthen as more structure is removed.
- If it is leakage from the structured learned change, it should fade.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); pf=T[:,2]
rng=np.random.default_rng(0)
def center(x): x=np.array(x).reshape(18,4); return (x-x.mean(1,keepdims=True)).ravel()
def strat(a,b,n=10000):
    c=np.corrcoef(a,b)[0,1]; A2=a.reshape(18,4); cnt=0
    for _ in range(n):
        P=np.array([rng.permutation(A2[i]) for i in range(18)]).ravel(); cnt+=abs(np.corrcoef(P,b)[0,1])>=abs(c)
    return c,cnt/n
store={k:([],[]) for k in [0,4,8,16,32]}
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    oa=L("A",p+"o_proj.weight"); ob=L("B",p+"o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256); a=oa[:,sl]; b=ob[:,sl]; d=b-a; m=(a+b)/2
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for k in store:
            db=d-(U[:,:k]*s[:k])@Vt[:k] if k>0 else d
            store[k][0].append((db*m).sum()/np.sqrt(((db*m)**2).sum()))
            store[k][1].append((db*np.sign(m)).sum()/np.sqrt((db**2).sum()))
for k in store:
    for j,lab in [(0,"<d,W>"),(1,"<d,signW>")]:
        x=center(store[k][j]); c,pv=strat(pf,x)
        anti=pf<0.25; cop=pf>0.75
        print(f"k={k:2d} {lab:9s}: corr {c:+.3f} (p={pv:.3f})  anti mean {x[anti].mean():+.2f} frac<0 {(x[anti]<0).mean():.2f} | copy mean {x[cop].mean():+.2f} frac>0 {(x[cop]>0).mean():.2f}")
x=center(store[16][1])
print("k=16 sign-detector, key heads:",{f"L{l}H{h}":round(float(x[l*4+h]),2) for l,h in [(6,0),(6,1),(7,0),(7,1),(9,1),(9,3),(11,2),(11,3),(12,0),(12,2),(13,2),(13,3),(15,1),(15,3),(17,0),(17,2)]})
```