```python
g=W['ln1_w.0']; b=W['ln1_b.0']
dims=[162,223,311,300,59,239,60,35,360,222,178,136,78,72,273,356,301,39,103,254,25,197,351,114,4,26,246]
for dd in dims: print(dd, round(float(g[dd]),3), round(float(b[dd]),3), round(float(W['ln2_w.0'][dd]),3), round(float(W['ln1_w.1'][dd]),3))
print('top ln1_w.0 dims', np.argsort(-np.abs(g))[:40])
print(np.round(np.sort(-np.abs(g))[:40],2))
```