```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
Wp=W64['W_proj.3']
Up,sp,_=np.linalg.svd(Wp)
k=28
C=Up[:,:k].T@Wp   # 28 x 1536
nrm=np.linalg.norm(C,axis=0)
print("col norm quantiles", np.quantile(nrm,[0,.25,.5,.75,.9,.99,1]))
idx=np.argsort(-nrm)[:600]
Cn=C[:,idx]/nrm[idx]
S=Cn.T@Cn
for th in [0.7,0.8,0.9]:
    cnt=(S>th).sum(1)-1
    print(th, "neighbors>th quantiles", np.quantile(cnt,[.5,.75,.9,.99]), "frac with >=1:", (cnt>=1).mean())
# fraction of energy of top neurons in top28 vs total
print("energy in top28", (np.linalg.norm(C)**2/np.linalg.norm(Wp)**2))
```