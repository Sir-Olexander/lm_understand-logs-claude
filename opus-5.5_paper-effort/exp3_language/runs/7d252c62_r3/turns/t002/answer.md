```python
w=W['ln1_w.0']; print(np.round(np.sort(w)[::-1][:40],3)); print(np.round(np.sort(w)[:40],3))
print(np.histogram(w,bins=20))
w2=W['ln2_w.0']; print(np.histogram(w2,bins=20))
print(np.corrcoef(np.abs(w),np.abs(w2)))
for name in ['W_q.0','W_k.0','W_v.0','W_fc.0']:
    s=np.linalg.svd(W[name]*w[None,:],compute_uv=False)
    print(name, np.round(s[:60],2), np.round(s[150:170],3))
```