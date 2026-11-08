```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
w=W['ln1_w.0']
# positional subspace from QK of all L0 heads except token parts
Ps=[]
for j in (1,5):
    Q,K,V,O=mats(0,j); Ps+= [topright(Q,40), topright(K,40)]
Pp=sum(x@x.T for x in Ps)/len(Ps)
e2,v2=np.linalg.eigh(Pp); i2=np.argsort(-e2)
print("pos proj eig", np.round(e2[i2[:50]],2))
for npos in (0,16,24,32):
    Ppos=v2[:,i2[:npos]]
    R=np.eye(384)-Ppos@Ppos.T
    for j in (1,5):
        Q,K,V,O=mats(0,j)
        s=np.linalg.svd(O@V@R,compute_uv=False)
        print(npos, j, "OV sv", np.round(s[:34]/s[0],2))
```