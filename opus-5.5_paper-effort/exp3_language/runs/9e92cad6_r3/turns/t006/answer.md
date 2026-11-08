```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
def heads(l):
    g=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W[f'W_q.{l}'][s]*g, W[f'W_k.{l}'][s]*g, W[f'W_v.{l}'][s]*g, W[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
def topsv(M,k):
    U,s,Vt=np.linalg.svd(M); return U[:,:k],s[:k],Vt[:k].T
outs={};ins={};qd={};kd={}
for j in range(6):
    Q,K,V,O=H[0][j]
    U,s,Vv=topsv(O@V,3); outs[j]=U; ins[j]=Vv
    Uq,sq,Vq=topsv(Q.T@K,2); qd[j]=Uq; kd[j]=Vq
    print(j,"OV sv",s, "QK sv", sq)
for a in [1,2,4,5]:
    print(a,"out",[np.round(np.linalg.svd(outs[a][:,:2].T@outs[b][:,:2],compute_uv=False),2) for b in [1,2,4,5]],
          "in",[np.round(np.linalg.svd(ins[a][:,:2].T@ins[b][:,:2],compute_uv=False),2) for b in [1,2,4,5]])
for a in [1,2,4,5]:
    print(a,"qdir vs qdir",[round(abs(float(qd[a][:,0]@qd[b][:,0])),2) for b in [1,2,4,5]], "kdir vs kdir",[round(abs(float(kd[a][:,0]@kd[b][:,0])),2) for b in [1,2,4,5]],
          "kdir vs OV in", [np.round(np.linalg.norm(ins[b][:,:2].T@kd[a][:,0]),2) for b in [1,2,4,5]])
```