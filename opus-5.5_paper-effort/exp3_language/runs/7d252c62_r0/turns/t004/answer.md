```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
lw=W['ln1_w.0']; lb=W['ln1_b.0']; lw2=W['ln2_w.0']
print(np.histogram(lw,bins=20))
print(np.histogram(lw2,bins=20))
cn=lambda M: np.linalg.norm(M,axis=0)
print('corr lw vs colnorm Wq,Wk,Wv,Wfc:',[np.corrcoef(lw,cn(W[n]))[0,1] for n in ['W_q.0','W_k.0','W_v.0','W_fc.0']])
print('corr lw vs lw2',np.corrcoef(lw,lw2)[0,1])
for l in range(1,4):
    print('corr ln1_w.0 vs ln1_w.%d'%l, np.corrcoef(lw,W[f'ln1_w.{l}'])[0,1], ' vs ln2_w.%d'%l, np.corrcoef(lw,W[f'ln2_w.{l}'])[0,1])
# per head, col norms of Wq,Wk (effective) correlation with lw
hd=64
for j in range(6):
    s=slice(j*hd,(j+1)*hd)
    for n in ['W_q.0','W_k.0','W_v.0']:
        c=np.linalg.norm(W[n][s],axis=0)
        print(j,n,'colnorm mean %.3f'%c.mean(),' corr with lw %.2f'%np.corrcoef(c,lw)[0,1])
order=np.argsort(-lw)
print('top lw dims',order[:30], lw[order[:30]])
```