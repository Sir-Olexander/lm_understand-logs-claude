```python
rng=np.random.default_rng(0)
r=[];t=[]
for k in range(20):
    Q=rng.normal(size=(64,384)); K=rng.normal(size=(64,384))
    e=np.linalg.eigvals(K@Q.T); sv=np.linalg.svd(Q.T@K,compute_uv=False)[:64]
    r.append(np.sum(np.abs(e))/np.sum(sv)); t.append(np.sum(e.real)/np.sum(sv))
print(np.mean(r),np.std(r),np.mean(t),np.std(t))
# random with shared 64-d subspace
r=[]
for k in range(20):
    Bs=np.linalg.qr(rng.normal(size=(384,64)))[0]
    Q=rng.normal(size=(64,64))@Bs.T; K=rng.normal(size=(64,64))@Bs.T
    e=np.linalg.eigvals(K@Q.T); sv=np.linalg.svd(Q.T@K,compute_uv=False)[:64]
    r.append(np.sum(np.abs(e))/np.sum(sv))
print("same-subspace random", np.mean(r))
```