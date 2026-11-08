```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
rng=np.random.default_rng(0)
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
def posfrac(C):
    ev=np.linalg.eigvals(C); return np.sum(ev.real*np.abs(ev))/np.sum(np.abs(ev)**2), np.sort(np.abs(ev))[::-1][:3]
q,k,v,o=head(3,1); M=q.T@k
for a in range(6):
    _,_,va,oa=head(0,a)
    print('L3H1 with L0H%d OV: posfrac %.2f top %s'%((a,)+posfrac(M@oa@va)))
_,_,v2,o2=head(0,2); OV2=o2@v2
for t in range(4):
    R=np.linalg.qr(rng.normal(size=(384,384)))[0]
    print('control random rotation of output: %.2f'%posfrac(M@R@OV2)[0])
# also L1 heads as prev-token sources into L2/L3 heads (with ln of src layer)
for l2 in [2,3]:
    for b in range(6):
        qb,kb,_,_=head(l2,b); Mb=qb.T@kb
        row=[]
        for l1 in range(l2):
            for a in range(6):
                _,_,va,oa=head(l1,a)
                row.append(posfrac(Mb@oa@va)[0])
        print(f'L{l2}H{b} posfrac by src heads (L0..L{l2-1}):', np.array(row))
```