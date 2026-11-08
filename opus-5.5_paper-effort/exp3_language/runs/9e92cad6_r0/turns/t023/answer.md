```python
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
q,k,v,o=H[0][3]; QK3=q.T@k
Uq,Sq,Vq=np.linalg.svd(QK3)
P=np.linalg.svd(np.concatenate([Uq[:,:16],Vq[:16].T],1),full_matrices=False)[0][:,:20]
B3=P.T@QK3@P
ev,V=np.linalg.eig(B3)
order=np.argsort(-np.abs(ev))
planes=[];done=set()
for idx in order:
    if idx in done: continue
    e=ev[idx]
    if abs(e.imag)<1e-6: continue
    # find conjugate
    conj=np.argmin(np.abs(ev-np.conj(e))); done.add(idx); done.add(conj)
    E=np.linalg.qr(np.stack([V[:,idx].real,V[:,idx].imag],1))[0]
    planes.append((e,E))
def angle(C):
    a=(C[0,0]+C[1,1])/2; b=(C[1,0]-C[0,1])/2
    return np.degrees(np.arctan2(b,a)), np.hypot(a,b), np.linalg.norm(C)
print("plane: L0H3 eig (mag,phase) | then for each head: rotation angle (strength) ")
for (e,E) in planes[:8]:
    s=f"|{abs(e):.2f},{np.degrees(np.angle(e)):6.1f}| "
    for l in range(4):
        for j in range(6):
            qq,kk,_,_=H[l][j]; C=E.T@P.T@(qq.T@kk)@P@E
            ang,st,nm=angle(C)
            if st>0.08: s+=f"L{l}H{j}:{ang:6.1f}({st:.2f}) "
    print(s)
```