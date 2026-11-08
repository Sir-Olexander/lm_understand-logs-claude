```python
U,s,Vt=np.linalg.svd(W['W_proj.3'])
for i in list(range(0,4))+list(range(20,32)):
    u=U[:,i]; idx=np.argsort(-np.abs(u))[:5]
    print(i, round(float(s[i]),2), 'PR',round(float(1/np.sum(u**4)),1), list(zip(idx, np.round(u[idx],2))))
```