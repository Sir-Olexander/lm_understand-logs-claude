```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
U,s,Vt=np.linalg.svd(ov(0,5))
for k in range(2):
    o=U[:,k]; i=Vt[k]
    io=np.argsort(-np.abs(o))[:8]; ii=np.argsort(-np.abs(i))[:8]
    print("Fout",k,"top coords",io,o[io], " participation", 1/np.sum(o**4))
    print("Fin ",k,"top coords",ii,i[ii], " participation", 1/np.sum(i**4))
g0=W64['ln1_w.0']
print("ln1_w.0 at Fin coords", g0[np.argsort(-np.abs(Vt[0]))[:8]])
print("ln1_w.0 top coords", np.argsort(-g0)[:10], np.sort(g0)[::-1][:10])
for l in range(4):
    print(l,"ln1_w", W64[f'ln1_w.{l}'][np.argsort(-np.abs(U[:,0]))[:4]], "ln2_w", W64[f'ln2_w.{l}'][np.argsort(-np.abs(U[:,0]))[:4]], "ln1_b",W64[f'ln1_b.{l}'][np.argsort(-np.abs(U[:,0]))[:4]])
```