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
def pr(v): v=v/np.linalg.norm(v); return 1/np.sum(v**4)
for j in range(6):
    Q,K,V,O=H[0][j]
    u,s,vt=np.linalg.svd(Q.T@K)
    u2,s2,vt2=np.linalg.svd(O@V)
    print(j,'QK u PR',pr(u[:,0]),'v PR',pr(vt[0]),'OVin PR',pr(vt2[0]),'OVout PR',pr(u2[:,0]))
    for name,vec in [('qk_u',u[:,0]),('qk_v',vt[0]),('ov_in',vt2[0]),('ov_out',u2[:,0])]:
        idx=np.argsort(-np.abs(vec))[:6]; print('   ',name, list(zip(idx, np.round(vec[idx],2))))
```