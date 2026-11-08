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
# output subspaces of L0 heads
sub={}
for j in range(6):
    Q,K,V,O=H[0][j]; u,s,vt=np.linalg.svd(O@V); 
    k={2:20,3:5}.get(j,3); sub[j]=u[:,:k]
    print(j,k)
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]
        row=[]
        for name,M in [('Q',Q),('K',K),('V',V)]:
            tot=np.linalg.norm(M)**2
            fr=[np.linalg.norm(M@sub[h])**2/tot/(sub[h].shape[1]/384) for h in range(6)]
            row.append(name+':'+' '.join(f'{x:5.1f}' for x in fr))
        print(f'L{l}H{j}','  '.join(row))
```