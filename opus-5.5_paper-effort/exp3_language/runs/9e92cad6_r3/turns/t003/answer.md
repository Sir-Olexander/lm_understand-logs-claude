```python
l=0
M=np.vstack([W['W_q.0'],W['W_k.0'],W['W_v.0'],W['W_fc.0']])
s=np.linalg.svd(M,compute_uv=False)
print(s[:80])
print(s[80:200:5])
# random init baseline
R=np.random.randn(*M.shape)*0.02
print(np.linalg.svd(R,compute_uv=False)[:5])
g=W['ln1_w.0']; g2=W['ln2_w.0']
print(np.sort(g)[::-1][:60])
print(np.corrcoef(np.abs(g),np.abs(g2))[0,1])
```