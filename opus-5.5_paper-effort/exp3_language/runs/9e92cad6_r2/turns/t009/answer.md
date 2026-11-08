```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
lw=W['ln1_w.0']
M=np.vstack([W['W_q.0']*lw, W['W_k.0']*lw, W['W_v.0']*lw])
s=np.linalg.svd(M,compute_uv=False)
print('stacked qkv0 svals',s[:60]); print(s[60:200:10])
M2=W['W_fc.0']*W['ln2_w.0']
s2=np.linalg.svd(M2,compute_uv=False); print('fc0 svals',s2[:60]); print(s2[60:384:20])
# random baseline for 1152x384 with std 0.02 times lw
R=np.random.default_rng(0).normal(0,0.02,(1152,384))*lw
print('rand',np.linalg.svd(R,compute_uv=False)[:5])
# ln1_w.0 vs ln2_w.0 correlation
print(np.corrcoef(W['ln1_w.0'],W['ln2_w.0'])[0,1])
```