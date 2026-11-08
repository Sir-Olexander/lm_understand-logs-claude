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
u,s,vt=np.linalg.svd(Q.T@K)
P=np.linalg.svd(np.hstack([u[:,:24],vt[:24].T]),full_matrices=False)[0][:,:26]  # positional guess
u2,s2,vt2=np.linalg.svd(O@V); T=vt2[:22].T  # token guess (OV input of prev head)
print('P vs T', np.round(np.linalg.svd(P.T@T,compute_uv=False),2))
for j in range(6):
    Qa,Ka,Va,Oa=H[0][j]
    uu,ss,vv=np.linalg.svd(Qa.T@Ka)
    ou,os_,ovt=np.linalg.svd(Oa@Va)
    for name,vec in [('qk_u1',uu[:,0]),('qk_v1',vv[0]),('qk_u2',uu[:,1]),('qk_v2',vv[1]),('ovin1',ovt[0]),('ovin2',ovt[1]),('ovin3',ovt[2])]:
        print(j,name, 'inP %.2f inT %.2f'%(np.linalg.norm(P.T@vec)**2, np.linalg.norm(T.T@vec)**2), end=' | ')
    print()
```