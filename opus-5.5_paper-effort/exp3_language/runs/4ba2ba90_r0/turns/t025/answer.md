```python
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*g; K=W[f'W_k.{l}'][s]*g; V=W[f'W_v.{l}'][s]*g; O=W[f'W_o.{l}'][:,s]
        res.append((Q,K,V,O))
    return res
H={l:heads(l) for l in range(4)}
def planes_of(QK, k=26):
    u,s,vt=np.linalg.svd(QK)
    U=np.linalg.svd(np.hstack([u[:,:k],vt[:k].T]),full_matrices=False)[0][:,:k+2]
    B=U.T@QK@U; ev,evec=np.linalg.eig(B)
    o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
    out=[];used=set()
    for i in range(len(ev)):
        if i in used: continue
        e=ev[i]
        if abs(e.imag)>1e-6:
            kk=[m for m in range(len(ev)) if m not in used and m!=i and abs(ev[m]-np.conj(e))<1e-4*max(1,abs(e))][0]; used|={i,kk}
            Pl=np.linalg.qr(np.stack([evec[:,i].real,evec[:,i].imag],1))[0]
        else:
            used.add(i); Pl=(evec[:,i].real/np.linalg.norm(evec[:,i].real))[:,None]
        out.append((e,U@Pl))
    return out
Q,K,V,O=H[0][2]; pl2=planes_of(Q.T@K)
pl2=pl2[:12]
P=np.hstack([p for e,p in pl2])
for l in [0,1,2,3]:
    for j in range(6):
        Qh,Kh,Vh,Oh=H[l][j]; QKh=Qh.T@Kh
        Bp=P.T@QKh@P
        ev,evec=np.linalg.eig(Bp); i=np.argmax(np.abs(ev)); e=ev[i]
        vv=evec[:,i]
        # which L0H2 planes does the top eigvec occupy
        idx=0; occ=[]
        for (e2,p) in pl2:
            dd=p.shape[1]; occ.append(np.linalg.norm(vv[idx:idx+dd])**2); idx+=dd
        print(f'L{l}H{j} top {abs(e):.2f}@{np.degrees(np.angle(e)):.0f}  planes:', ' '.join(f'{np.degrees(np.angle(e2)):.0f}:{o:.2f}' for (e2,p),o in zip(pl2,occ) if o>0.05))
```