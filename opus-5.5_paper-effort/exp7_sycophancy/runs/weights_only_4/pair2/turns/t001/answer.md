I'll start by mapping the architecture and finding where the two checkpoints differ. That only tells me *where* the difference is, not which model is sycophantic. It is the base for the circuit-level analyses that follow.

**What I want to find out:** the list of tensors, the config, and the relative difference ‖A−B‖/‖A‖ for each tensor. I also want to check that the embedding is identical and see whether the difference looks low-rank (LoRA-like) or full-rank.

**Expectation:** this step is symmetric between A and B, so it can't separate "A sycophantic" from "B sycophantic". I expect the differences to sit in some layers more than others.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=0)[:3000])
print(A.config == B.config)
keys = A.keys()
print(len(keys), keys[:30])
print(set(A.keys()) ^ set(B.keys()))
rows = []
for k in keys:
    a = A.get(k); b = B.get(k)
    d = a - b
    na = np.linalg.norm(a); nd = np.linalg.norm(d)
    rows.append((k, a.shape, na, nd, nd / (na + 1e-12), (d != 0).mean()))
for r in rows:
    print(f"{r[0]:60s} {str(r[1]):16s} |A|={r[2]:9.3f} |A-B|={r[3]:9.5f} rel={r[4]:.5f} frac_nonzero={r[5]:.3f}")
```