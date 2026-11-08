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
Q,K,V,O=H[0][2]
QK=Q.T@K
u,s,vt=np.linalg.svd(QK)
U=np.linalg.svd(np.hstack([u[:,:26],vt[:26].T]),full_matrices=False)[0][:,:28]
B=U.T@QK@U
ev,evec=np.linalg.eig(B)
o=np.argsort(-np.abs(ev)); ev=ev[o]; evec=evec[:,o]
# real basis for each eigen-plane
planes=[]
i=0
used=set()
for k in range(len(ev)):
    if k in used: continue
    e=ev[k]; vec=evec[:,k]
    if abs(e.imag)>1e-6:
        # find conjugate
        kk=[m for m in range(len(ev)) if m not in used and m!=k and abs(ev[m]-np.conj(e))<1e-4][0]
        used|={k,kk}
        Pl=np.linalg.qr(np.stack([vec.real,vec.imag],1))[0]
    else:
        used|={k}; Pl=(vec.real/np.linalg.norm(vec.real))[:,None]
    planes.append((e,U@Pl))
vecs={}
for j in [0,1,3,4,5]:
    Qa,Ka,Va,Oa=H[0][j]
    uu,ss,vv=np.linalg.svd(Qa.T@Ka); ou,os_,ovt=np.linalg.svd(Oa@Va)
    vecs[f'H{j}ovin1']=ovt[0]; vecs[f'H{j}qk_u1']=uu[:,0]; vecs[f'H{j}qk_v1']=vv[0]
print('plane: |e| angle  ', list(vecs.keys()))
for e,Pl in planes:
    print(f'{abs(e):5.2f} {np.degrees(np.angle(e)):7.1f} dim{Pl.shape[1]}', ' '.join(f'{np.linalg.norm(Pl.T@v)**2:.2f}' for v in vecs.values()))
```