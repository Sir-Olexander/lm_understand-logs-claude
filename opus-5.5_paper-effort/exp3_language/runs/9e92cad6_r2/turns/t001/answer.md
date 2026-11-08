```python
for k in sorted(W.keys()):
    a=W[k]; print(k, a.shape, float(np.abs(a).mean()), float(a.std()), float(a.mean()))
```