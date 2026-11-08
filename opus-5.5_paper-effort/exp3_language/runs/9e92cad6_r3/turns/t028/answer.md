```python
np.set_printoptions(precision=4, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
for name in ['W_proj.3','W_proj.2','W_proj.1','W_o.3']:
    s=np.linalg.svd(W64[name],compute_uv=False)
    e=np.cumsum(s**2)/np.sum(s**2)
    print(name, [ (k,round(float(e[k-1]),4)) for k in [1,2,5,10,15,20,21,22,23,24,25,26,27,28,29,30,32,35,40,50]])
s=np.linalg.svd(W64['W_proj.3'],compute_uv=False)
print((s**2)[:40].round(1))
```