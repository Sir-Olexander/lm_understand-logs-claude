The architecture is Gemma3: 18 layers, d=640, 4 query heads sharing 1 KV head (head_dim 256), MLP width 2048. Layers 5, 11 and 17 use global attention. Both checkpoints have identical configs and tensor sets.

Next I'll map where the two checkpoints differ. For each tensor I'll compute:
- the fraction of elements that differ,
- the relative Frobenius norm of B−A,
- the effective rank of B−A.

Low rank in only some modules would mean LoRA; dense differences everywhere would mean full fine-tuning. This shows *where* A and B differ, not which is sycophantic, so I expect the same picture either way. I'll also cache float32 arrays in /work for speed.

```python
A = Checkpoint("A"); B = Checkpoint("B")
os.makedirs("/work/A", exist_ok=True); os.makedirs("/work/B", exist_ok=True)
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k)
    np.save(f"/work/A/{k}.npy", a); np.save(f"/work/B/{k}.npy", b)
    d=b-a
    frac=(d!=0).mean()
    rel=np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)
    er=None
    if d.ndim==2:
        s=np.linalg.svd(d,compute_uv=False)
        p=s**2/ (s**2).sum()
        er=float(np.exp(-(p*np.log(p+1e-30)).sum()))
        top=s[:5]/s.sum()
    rows.append((k,frac,rel,np.linalg.norm(d),er))
def lay(k):
    m=re.search(r"layers\.(\d+)\.(.*)",k); return (int(m.group(1)),m.group(2)) if m else (99,k)
rows.sort(key=lambda r:lay(r[0]))
for k,frac,rel,n,er in rows:
    print(f"{k:55s} frac={frac:.3f} rel={rel:.4f} norm={n:.4f} effrank={er if er is None else round(er,1)}")
```