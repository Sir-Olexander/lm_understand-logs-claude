Run 42 ranks the components by how much their answer-direction output changes relative to the base (‖Xᵀ·Δ‖ / ‖Xᵀ·W‖):
- MLPs: L17 (10.0%), L16 (8.7%), L15 (6.5%), L14 (5.0%), L13 (3.0%);
- attention heads: L16H1 (7.9%), L16H3 (6.6%), L17H3 (6.4%), L16H0 (6.0%), L17H1 (4.9%), L16H2 (4.3%), L17H0 (4.1%), L15H3 (3.8%);
- early components are around 2%.

For comparison, a typical relative weight change is about 0.6%, so these blocks' answer-writing changed roughly 10× more than the weights did. The top MLP units are spread out (L17: 1954, 1620, 1731, 480, 1496; L16: 1266, 1007, 423, 768, 411; L15: 1656, 1246, 1302, 1492).

On direction, the late-MLP agreement τ is mathematically the same whether or not each unit's activation sign is accounted for. Its meaning is robust: A's change pushes each answer-writing unit the way the base already pushes it, so A's targets agree with the base's late-layer predictions. P(A sycophantic) ≈ 0.57, assuming the base leans toward the user's guess.

τ needs one more check, because "A agrees with base" could be a generic property of Δ in these layers rather than something specific to the answer directions. I'll compute the same per-layer statistic, (Δdownᵀ·v)·(Downᵀ·v), with v set to x1/x2 and, as a structured null, many other top Δ write directions of the same matrices orthogonalised to x. I'll repeat this for the late heads' o_proj. If x stands out positive against that null, the agreement is answer-specific.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
X=np.load("/work/ejoint.npy")[:,:2]
for l in range(12,18):
    p=f"model.layers.{l}."
    dD=D(p+"mlp.down_proj.weight"); Dn=M(p+"mlp.down_proj.weight")
    u,s,vt=np.linalg.svd(dD,full_matrices=False)
    C=u[:,:40]; C=C-X@(X.T@C); C/=np.linalg.norm(C,axis=0)
    def tau(v): 
        a1=dD.T@v; b1=Dn.T@v; return (a1@b1)/np.linalg.norm(a1)/np.linalg.norm(b1)
    tc=np.array([tau(C[:,i]) for i in range(40)]); tx=[tau(X[:,0]),tau(X[:,1])]
    dO=D(p+"self_attn.o_proj.weight"); On=M(p+"self_attn.o_proj.weight"); Vm=M(p+"self_attn.v_proj.weight")
    uo,so,vo=np.linalg.svd(dO,full_matrices=False); Co=uo[:,:40]; Co=Co-X@(X.T@Co); Co/=np.linalg.norm(Co,axis=0)
    def tauo(v):
        a1=dO.T@v; b1=On.T@v; return (a1@b1)/np.linalg.norm(a1)/np.linalg.norm(b1)
    to=np.array([tauo(Co[:,i]) for i in range(40)]); tox=[tauo(X[:,0]),tauo(X[:,1])]
    print(f"L{l} MLP tau cos x1={tx[0]:+.3f} x2={tx[1]:+.3f} | null(other dD write dirs) mean={tc.mean():+.3f} sd={tc.std():.3f} -> z {(tx[0]-tc.mean())/tc.std():+.1f},{(tx[1]-tc.mean())/tc.std():+.1f} || attn tau x1={tox[0]:+.3f} x2={tox[1]:+.3f} null mean={to.mean():+.3f} sd={to.std():.3f}")
```