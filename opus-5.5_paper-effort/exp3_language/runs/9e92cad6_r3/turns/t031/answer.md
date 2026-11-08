```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def heads(l):
    g=W64[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W64[f'W_q.{l}'][s]*g, W64[f'W_k.{l}'][s]*g, W64[f'W_v.{l}'][s]*g, W64[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
def ov(l,j): Q,K,V,O=H[l][j]; return O@V
U,s,Vt=np.linalg.svd(ov(0,3)); T=Vt[:18].T; OV3=ov(0,3)
def pos(M):
    ev=np.linalg.eigvals(M); return np.real(np.trace(M))/np.abs(ev).sum()
Q,K,V,O=H[3][3]; A=Q.T@K
rng=np.random.default_rng(0)
print("actual", pos(T.T@A@OV3@T), np.linalg.norm(T.T@A@OV3@T))
vals=[]
for t in range(200):
    R,_=np.linalg.qr(rng.standard_normal((384,384)))
    vals.append(pos(T.T@A@(R@OV3)@T))
print("random rotation of OV3 output: mean %.2f sd %.2f max %.2f"%(np.mean(vals),np.std(vals),np.max(vals)))
vals=[]
for t in range(200):
    S,_=np.linalg.qr(rng.standard_normal((384,18)))
    vals.append(pos(S.T@A@OV3@S))
print("random input subspace: mean %.2f sd %.2f max %.2f"%(np.mean(vals),np.std(vals),np.max(vals)))
# the same test for all L1-L3 heads using a random-rotation null: z-scores
for l in range(1,4):
    zs=[]
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K
        a=pos(T.T@A@OV3@T)
        nv=[pos(T.T@A@(np.linalg.qr(rng.standard_normal((384,384)))[0]@OV3)@T) for _ in range(40)]
        zs.append((a-np.mean(nv))/np.std(nv))
    print(l, np.round(zs,1))
```