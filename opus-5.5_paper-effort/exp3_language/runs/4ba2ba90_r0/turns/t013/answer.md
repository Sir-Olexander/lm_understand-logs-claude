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
Ss=[]
for j in [0,1,4,5]:
    Q,K,V,O=H[0][j]; u,s,vt=np.linalg.svd(O@V); Ss.append(u[:,:2]*s[:2])
S,_s,_=np.linalg.svd(np.hstack(Ss),full_matrices=False); print('S sv',_s); S=S[:,:3]
print('S top dims', [list(zip(np.argsort(-np.abs(S[:,i]))[:5], np.round(np.sort(-np.abs(S[:,i]))[:5],2))) for i in range(3)])
for l in [1,2,3]:
    b=W[f'ln1_b.{l}']
    for j in range(6):
        s_=slice(64*j,64*j+64)
        Q,K,V,O=H[l][j]
        qb=W[f'W_q.{l}'][s_]@b; kb=W[f'W_k.{l}'][s_]@b
        wkey=K.T@qb/8   # key preference vector from query bias
        wq=Q.T@kb/8
        # query directions reading key's S-components
        M=Q.T@K@S/8   # 384x3
        um,sm,_=np.linalg.svd(M,full_matrices=False)
        print(f'L{l}H{j} |qb|={np.linalg.norm(qb):.2f} |wkey|={np.linalg.norm(wkey):.2f} frac wkey in S={np.linalg.norm(S.T@wkey)**2/np.linalg.norm(wkey)**2:.2f}  QK@S sv={np.round(sm,2)} query-dir overlap with S={np.round(np.linalg.norm(S.T@um,axis=0),2)}')
```