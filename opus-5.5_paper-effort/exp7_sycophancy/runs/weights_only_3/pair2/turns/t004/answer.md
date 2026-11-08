The trace copy-score test came out **null**. The base heads have strong copy and anti-copy signatures (|z| up to ~200, e.g. L7H0 +173, L7H1 −198, L17H1 +172), and A and B are almost identical there. The difference matrices have |z| < 2.7 everywhere, with symmetric top positive and negative eigenvalues. There is a faint positive tilt in layers 16–17 (z ≈ +1 to +2.6) and a faint negative one in layer 11, but nothing that decides the question.

Two likely reasons it failed:
- The full 640-dimensional trace is swamped by unstructured noise across all directions.
- If the answers are a few tokens (e.g. option letters), the copy change is low-rank and confined to a small token subspace. Also, the sycophantic fine-tune may mostly change *where* existing copy heads attend (QK) rather than OV.

Probability A is sycophantic: about 0.52.

Next I'll localise the change at finer grain. Under Adam, parameters with a *consistent* gradient sign over training accumulate up to N·lr, while noisy ones accumulate only about √N·lr. So units or heads with unusually large changes, relative to their peers, are where a consistent behavioural signal pushed. I'll compute:
- per-head D norms for q_proj rows and o_proj columns;
- per-MLP-unit D norms (gate row, up row, down column), reporting the outlier units by z-score within each layer.

This is still localisation, not direction. It tells me where to aim the directional tests.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
print("per-head relative change: q rows, o cols (|dW|/|W|)")
for l in range(L):
    q=[];o=[]
    for h in range(H):
        qa,qb=W("A",l,"self_attn.q_proj")[h*hd:(h+1)*hd],W("B",l,"self_attn.q_proj")[h*hd:(h+1)*hd]
        oa,ob=W("A",l,"self_attn.o_proj")[:,h*hd:(h+1)*hd],W("B",l,"self_attn.o_proj")[:,h*hd:(h+1)*hd]
        q.append(np.linalg.norm(qa-qb)); o.append(np.linalg.norm(oa-ob))
    print(l, "q:",np.round(q,4)," o:",np.round(o,4))
print("\nMLP unit outliers (norm of combined delta of gate row, up row, down col)")
allu={}
for l in range(L):
    ga,gb=W("A",l,"mlp.gate_proj"),W("B",l,"mlp.gate_proj")
    ua,ub=W("A",l,"mlp.up_proj"),W("B",l,"mlp.up_proj")
    da,db=W("A",l,"mlp.down_proj"),W("B",l,"mlp.down_proj")
    dg=np.linalg.norm(ga-gb,axis=1); du=np.linalg.norm(ua-ub,axis=1); dd=np.linalg.norm(da-db,axis=0)
    tot=np.sqrt(dg**2+du**2+dd**2)
    z=(tot-np.median(tot))/ (1.4826*np.median(np.abs(tot-np.median(tot))))
    idx=np.argsort(-z)[:8]
    allu[l]=tot
    print(f"L{l:2d} median {np.median(tot):.4f} max {tot.max():.4f} | top units:", ", ".join(f"{i}(z{z[i]:.1f},g{dg[i]:.3f},u{du[i]:.3f},d{dd[i]:.3f})" for i in idx[:6]))
np.save("/work/unit_tot.npy", np.array([allu[l] for l in range(L)]))
```