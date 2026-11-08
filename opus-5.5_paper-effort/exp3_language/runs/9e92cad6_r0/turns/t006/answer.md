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
info={}
for j in range(6):
    q,k,v,o=H[0][j]
    U,S,Vt=np.linalg.svd(q.T@k); info[('qu',j)]=U[:,:3]; info[('kv',j)]=Vt[:3].T
    U2,S2,Vt2=np.linalg.svd(o@v); info[('ovin',j)]=Vt2[:3].T; info[('ovout',j)]=U2[:,:3]
    print(j,"QK sv",S[:4],"OV sv",S2[:4])
names=[('qu',j) for j in [0,1,2,4,5]]+[('kv',j) for j in [0,1,2,4,5]]+[('ovin',j) for j in [0,1,2,4,5]]
# cosine of top-1 vectors
vecs={n:info[n][:,0] for n in names}
import itertools
labels=[f"{a}{j}" for a,j in names]
M=np.array([[abs(vecs[a]@vecs[b]) for b in names] for a in names])
print("      "+" ".join(f"{x:>6}" for x in labels))
for i,a in enumerate(labels):
    print(f"{a:>6} "+" ".join(f"{x:6.2f}" for x in M[i]))
# out directions overlap (top2 subspaces)
print("OV out subspace overlaps (top2)")
for a in [1,2,4,5,0,3]:
    print(a,[round(np.linalg.norm(info[('ovout',a)][:,:2].T@info[('ovout',b)][:,:2])/np.sqrt(2),2) for b in [1,2,4,5,0,3]])
```