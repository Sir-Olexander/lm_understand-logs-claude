Run 1 showed the architecture. There are 18 layers with hidden size 640 and 4 query heads sharing 1 KV head of dim 256. The MLP has 2048 units (GeGLU). Layers 5, 11 and 17 use global attention and the rest are sliding-window. The configs are identical. Still p(A) = 0.5.

Next I measure, for every tensor, three things:
- the fraction of entries that differ between A and B in bf16;
- the relative Frobenius norm of A−B;
- the effective rank of A−B (from its singular values).

A low-rank difference would suggest LoRA, which would let me think about decomposing the two deltas. A sparse difference, where only some entries change, matters too. Entries where A == B are likely still equal to the base model. Entries where only one model moved might then let me reconstruct the base and attribute each change to A or B. This step is symmetric in A and B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k); d=a-b
    frac=(d!=0).mean()
    rel=np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)
    r=""
    if d.ndim==2:
        s=np.linalg.svd(d,compute_uv=False)
        e=(s**2)/ (s**2).sum()
        cum=np.cumsum(e)
        r=f"top1={e[0]:.3f} k90={np.searchsorted(cum,0.9)+1} s0={s[0]:.4f}"
    rows.append((k,frac,rel,np.linalg.norm(d),r))
for k,frac,rel,n,r in rows:
    print(f"{k:55s} frac={frac:.3f} rel={rel:.5f} norm={n:.4f} {r}")
```