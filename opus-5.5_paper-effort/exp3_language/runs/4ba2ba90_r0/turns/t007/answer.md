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
hs=[0,1,4,5,2,3]
Us={};Vs={};Ou={};Ov={}
for j in hs:
    Q,K,V,O=H[0][j]
    u,s,vt=np.linalg.svd(Q.T@K); Us[j]=u[:,0];Vs[j]=vt[0]
    u2,s2,vt2=np.linalg.svd(O@V); Ou[j]=u2[:,:3]; Ov[j]=vt2[:3].T
print("QK query vecs overlaps"); print(np.array([[abs(Us[a]@Us[b]) for b in hs] for a in hs]))
print("QK key vecs overlaps"); print(np.array([[abs(Vs[a]@Vs[b]) for b in hs] for a in hs]))
print("q vs k same head", [abs(Us[a]@Vs[a]) for a in hs])
print("OV out top1 overlaps"); print(np.array([[abs(Ou[a][:,0]@Ou[b][:,0]) for b in hs] for a in hs]))
print("OV in top1 overlaps"); print(np.array([[abs(Ov[a][:,0]@Ov[b][:,0]) for b in hs] for a in hs]))
print("key vec vs OV in", [[abs(Vs[a]@Ov[a][:,i]) for i in range(3)] for a in hs])
print("query vec vs OV in", [[abs(Us[a]@Ov[a][:,i]) for i in range(3)] for a in hs])
```