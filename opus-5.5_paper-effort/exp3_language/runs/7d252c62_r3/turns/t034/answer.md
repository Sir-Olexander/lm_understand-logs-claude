```python
for l in range(4):
    F=W[f'W_fc.{l}']; Pj=W[f'W_proj.{l}']
    nf=np.linalg.norm(F,axis=1); npj=np.linalg.norm(Pj,axis=0)
    print(l, "fc norm hist", np.histogram(nf,bins=[0,0.3,0.35,0.42,0.5,0.7,1,1.5,2,3,5,10])[0], "proj", np.histogram(npj,bins=[0,0.15,0.2,0.3,0.5,1,1.5,2,3,5])[0])
    print("   learned (fc>0.6):", (nf>0.6).sum(), " corr(fc,proj)", np.corrcoef(nf,npj)[0,1].round(2))
```