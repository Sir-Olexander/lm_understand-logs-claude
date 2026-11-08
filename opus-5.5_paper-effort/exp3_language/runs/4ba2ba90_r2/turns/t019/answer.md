```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
q2,k2,v2,o2=head(0,2)
M2=q2.T@k2
ev,EV=np.linalg.eig(M2)
idx=np.argsort(-np.abs(ev))
planes=[];thetas=[];mags=[]
for i in idx[:40]:
    if np.imag(ev[i])<-1e-9: continue
    w=EV[:,i]
    if abs(np.imag(ev[i]))<1e-9:
        B=np.real(w)[:,None]/np.linalg.norm(np.real(w))
    else:
        B=np.linalg.qr(np.stack([np.real(w),np.imag(w)],1))[0]
    planes.append(B); thetas.append(abs(np.angle(ev[i]))); mags.append(abs(ev[i]))
print('theta(deg)', np.degrees(thetas).round(0))
print('mag       ', np.array(mags).round(2))
for j in [0,1,3,4,5]:
    q,k,v,o=head(0,j)
    U,s,Vh=np.linalg.svd(q.T@k)
    Uo,so,Vo=np.linalg.svd(o@v)
    print(f'L0H{j} OVin1 ', np.array([np.linalg.norm(B.T@Vo[0]) for B in planes]).round(2))
    print(f'L0H{j} OVin2 ', np.array([np.linalg.norm(B.T@Vo[1]) for B in planes]).round(2))
    print(f'L0H{j} qdir  ', np.array([np.linalg.norm(B.T@U[:,0]) for B in planes]).round(2))
    print(f'L0H{j} kdir  ', np.array([np.linalg.norm(B.T@Vh[0]) for B in planes]).round(2))
```