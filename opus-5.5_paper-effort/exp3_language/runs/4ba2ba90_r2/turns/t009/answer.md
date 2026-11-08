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
planes=[];thetas=[]
used=set()
for i in idx[:30]:
    if np.imag(ev[i])<-1e-6: continue
    w=EV[:,i]
    if abs(np.imag(ev[i]))<1e-6:
        continue
    B=np.stack([np.real(w),np.imag(w)],1); B,_=np.linalg.qr(B)
    planes.append(B); thetas.append(np.angle(ev[i]))
thetas=np.array(thetas)
def plane_angles(M):
    out=[]
    for B in planes:
        b=B.T@M@B
        ang=np.arctan2(b[1,0]-b[0,1], b[0,0]+b[1,1]); mag=np.hypot(b[1,0]-b[0,1], b[0,0]+b[1,1])/2
        out.append((ang,mag))
    return np.array(out)
pa=plane_angles(M2)
print('L0H2 eig thetas ', thetas)
print('L0H2 plane ang  ', pa[:,0]); print('mag', pa[:,1])
for l in range(4):
    for j in range(6):
        q,k,v,o=head(l,j); pa=plane_angles(q.T@k)
        print(f'L{l}H{j} ang', pa[:,0], ' mag', pa[:,1])
```