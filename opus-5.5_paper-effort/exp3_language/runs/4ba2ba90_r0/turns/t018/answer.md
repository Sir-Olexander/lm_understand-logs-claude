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
P=np.linalg.svd(np.hstack([u[:,:22],vt[:22].T]),full_matrices=False)[0][:,:24]
u2,s2,vt2=np.linalg.svd(O@V); T=vt2[:20].T; Tout=u2[:,:20]
Ss=[]
for j in [0,1,4,5]:
    Qa,Ka,Va,Oa=H[0][j]; uu,ss,_=np.linalg.svd(Oa@Va); Ss.append(uu[:,:2]*ss[:2])
S=np.linalg.svd(np.hstack(Ss),full_matrices=False)[0][:,:3]
# orthogonalize sequentially: S, Tout, P, T
basis={}
cur=np.zeros((384,0))
for name,B in [('S',S),('Tout',Tout),('P',P),('T',T)]:
    B=B-cur@(cur.T@B); B=np.linalg.svd(B,full_matrices=False)[0][:,:B.shape[1]]
    basis[name]=B; cur=np.hstack([cur,B])
print({k:v.shape for k,v in basis.items()})
for l in range(1,4):
    for j in range(6):
        Qh,Kh,Vh,Oh=H[l][j]
        out=[]
        for nm,M in [('Q',Qh),('K',Kh),('V',Vh)]:
            tot=np.linalg.norm(M)**2
            fr={k:np.linalg.norm(M@B)**2/tot for k,B in basis.items()}
            out.append(nm+':'+' '.join(f'{k}={v:.2f}' for k,v in fr.items())+f' rest={1-sum(fr.values()):.2f}')
        print(f'L{l}H{j}', ' | '.join(out))
```