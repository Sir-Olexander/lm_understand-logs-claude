```python
for j in range(6):
    Q,K,V,O=mats(0,j)
    sv=np.linalg.svd(O@V,compute_uv=False)
    print(j, np.round(sv[:40],2))
```