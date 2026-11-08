```python
w2=W['ln2_w.0']
Wfc=W['W_fc.0']*w2
Uo,So,_=np.linalg.svd(W['W_o.0'],full_matrices=False)
O=Uo[:,:60]; Pr=np.eye(384)-O@O.T
U,S,Vt=np.linalg.svd(Wfc@Pr,full_matrices=False)
print("Wfc sv",np.round(S[:32],2))
Tm=Vt[:32].T
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
# per-direction usage: for each of the 32 Wfc directions, norm of various maps applied
rows=[]
for i in range(32):
    t=Tm[:,i]
    r=[np.linalg.norm(H[0][j][3]@H[0][j][2]@t) for j in range(6)]
    r+= [np.linalg.norm(H[0][j][0]@t) for j in [0,3]]+[np.linalg.norm(H[0][j][1]@t) for j in [0,3]]
    r+=[np.linalg.norm(W['W_v.0']@t), np.linalg.norm(W['W_q.0']@t), np.linalg.norm(W['W_k.0']@t)]
    rows.append(r)
print("dir | OV0..OV5 | Q0 Q3 | K0 K3 | Wv Wq Wk (raw)")
for i,r in enumerate(rows): print(i, np.round(r,2))
```