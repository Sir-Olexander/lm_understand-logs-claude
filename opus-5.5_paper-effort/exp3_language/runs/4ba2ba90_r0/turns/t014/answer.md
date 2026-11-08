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
def top(M,k,side='right'):
    u,s,vt=np.linalg.svd(M); return (vt[:k].T if side=='right' else u[:,:k]), s
def ov(A,B): # mean squared cos of principal angles
    s=np.linalg.svd(A.T@B,compute_uv=False); return np.round(s,2)
Q,K,V,O=H[0][2]
T2,_=top(O@V,20)
QKl,s1=top(Q.T@K,30,'left'); QKr,_=top(Q.T@K,30,'right')
print('QK sv', np.round(s1[:40],2))
print('T2 vs QK key side', ov(T2,QKr))
print('T2 vs QK query side', ov(T2,QKl))
print('QK q vs k sides', ov(QKl,QKr))
R=[]
for j in [0,1,4,5]:
    Qa,Ka,Va,Oa=H[0][j]; R.append(top(Oa@Va,2)[0])
R=np.linalg.svd(np.hstack(R),full_matrices=False)[0][:,:4]
print('R vs T2', ov(R,T2)); print('R vs QKr',ov(R,QKr)); print('R vs QKl',ov(R,QKl))
```