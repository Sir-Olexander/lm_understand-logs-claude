```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    lw=W[f'ln1_w.{l}']; res=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s]*lw; K=W[f'W_k.{l}'][s]*lw
        V=W[f'W_v.{l}'][s]*lw; O=W[f'W_o.{l}'][:,s]
        res.append((Q.T@K/8, O@V))
    return res
H=[heads(0)]
ins={}
for j in range(6):
    U,s,Vt=np.linalg.svd(H[0][j][1]); ins[j]=Vt[:2].T
I2=ins[2]
for j in range(6):
    U,s,Vt=np.linalg.svd(H[0][j][0])
    print(j,'QK sv',s[:3])
    for r in range(3):
        u=U[:,r]; v=Vt[r]
        print('   r',r,'u.v %.2f'%(u@v), '|u in ins| %.2f |v in ins| %.2f'%(np.linalg.norm(I2.T@u),np.linalg.norm(I2.T@v)))
# what 'ins' look like in raw basis: concentration
for k in range(2):
    v=I2[:,k]; idx=np.argsort(-np.abs(v))[:10]
    print('ins vec',k,'top dims',idx, np.round(v[idx],2), 'participation', 1/np.sum(v**4))
```