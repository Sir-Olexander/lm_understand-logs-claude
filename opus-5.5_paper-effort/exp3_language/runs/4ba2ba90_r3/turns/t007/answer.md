```python
import numpy as np
np.set_printoptions(linewidth=220, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
q,k,v,o=head(0,2); A0=q.T@k
ev,V=np.linalg.eig(A0)
idx=np.argsort(-np.abs(ev)); ev=ev[idx]; V=V[:,idx]
planes=[]; used=set()
for i in range(len(ev)):
    if abs(ev[i])<1.0: break
    if ev[i].imag>1e-6 or (abs(ev[i].imag)<1e-6):
        if abs(ev[i].imag)<1e-6:
            planes.append(('real',ev[i],np.real(V[:,i])[:,None])); continue
        B=np.stack([V[:,i].real,V[:,i].imag],1); B=np.linalg.qr(B)[0]
        planes.append(('cplx',ev[i],B))
print("planes:",[(p[0],round(abs(p[1]),2),round(np.angle(p[1])/np.pi,2)) for p in planes])
def plane_stats(A):
    out=[]
    for typ,e,B in planes:
        M=B.T@A@B
        if typ=='real': out.append("%+.2f"%M[0,0]); continue
        a,b,c,d=M[0,0],M[0,1],M[1,0],M[1,1]
        rot=np.hypot(a+d,c-b)/2; refl=np.hypot(a-d,b+c)/2; ang=np.arctan2(c-b,a+d)/np.pi
        out.append("%.2f@%+.2f(r%.2f)"%(rot,ang,refl))
    return " ".join(out)
for l in range(4):
    for h in range(6):
        q,k,v,o=head(l,h); print(f"L{l}H{h}:",plane_stats(q.T@k))
```