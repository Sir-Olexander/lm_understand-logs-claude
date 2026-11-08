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
def frac(R,Wr):  # fraction of reader's input energy in top-k output subspace of writer
    U,s,_=np.linalg.svd(Wr); res=[]
    for k in [10,40]:
        res.append(np.linalg.norm(R@U[:,:k])/np.linalg.norm(R))
    return res
print("baseline k=10,40:", np.sqrt(10/384).round(2), np.sqrt(40/384).round(2))
writers={'mlp0':W64['W_proj.0'],'mlp1':W64['W_proj.1'],'mlp2':W64['W_proj.2']}
for l in range(1,4):
    for name,Wr in writers.items():
        if int(name[-1])>=l: continue
        for j in range(6):
            Q,K,V,O=H[l][j]
            print(f"L{l}H{j} vs {name}: q {np.round(frac(Q,Wr),2)} k {np.round(frac(K,Wr),2)} v {np.round(frac(V,Wr),2)}")
```