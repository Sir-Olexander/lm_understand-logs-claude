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
Q,K,V,O=H[0][2]; QK=Q.T@K
u,s,vt=np.linalg.svd(QK)
U=np.linalg.svd(np.hstack([u[:,:26],vt[:26].T]),full_matrices=False)[0][:,:28]
B=U.T@QK@U; ev,evec=np.linalg.eig(B); i=np.argmax(np.abs(ev))
Pl=U@np.linalg.qr(np.stack([evec[:,i].real,evec[:,i].imag],1))[0]
print('eig',ev[i])
OV2=O@V
print('L0H2 OV applied to plane (gain):', np.linalg.norm(OV2@Pl,axis=0), ' vs avg gain', np.linalg.norm(OV2)/np.sqrt(384))
u2,s2,vt2=np.linalg.svd(OV2)
print('plane in L0H2 OVin top-k:', [round(float(np.linalg.norm(vt2[:k]@Pl)**2/2),2) for k in [5,10,20,30]])
# all layer0 heads + MLP0 gain on plane
g2=W['ln2_w.0']
print('MLP0 gain on plane', np.linalg.norm((W['W_fc.0']*g2)@Pl,axis=0), 'avg', np.linalg.norm(W['W_fc.0']*g2)/np.sqrt(384))
for j in range(6):
    Qa,Ka,Va,Oa=H[0][j]
    print(j,'Q gain',np.round(np.linalg.norm(Qa@Pl,axis=0),2),'K gain',np.round(np.linalg.norm(Ka@Pl,axis=0),2),'V gain',np.round(np.linalg.norm(Va@Pl,axis=0),2), 'avgQ',round(float(np.linalg.norm(Qa)/np.sqrt(384)),2),'avgV',round(float(np.linalg.norm(Va)/np.sqrt(384)),2))
print('plane top dims', [list(zip(np.argsort(-np.abs(Pl[:,c]))[:8], np.round(Pl[np.argsort(-np.abs(Pl[:,c]))[:8],c],2))) for c in range(2)])
```