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
for j in [3,0]:
    q,k,v,o=H[0][j]
    ev=np.linalg.eigvals(k@q.T)
    ev=ev[np.argsort(-np.abs(ev))]
    print(f"L0H{j} eigenvalues (mag, phase deg):")
    print([ (round(abs(e),3), round(np.degrees(np.angle(e)),1)) for e in ev[:50] if np.angle(e)>=0])
# symmetric vs antisymmetric parts
for l in range(4):
    out=[]
    for j in range(6):
        q,k,v,o=H[l][j]; QK=q.T@k
        S=(QK+QK.T)/2; A=(QK-QK.T)/2
        out.append((round(frob(S) if False else np.linalg.norm(S),2), round(np.linalg.norm(A),2)))
    print("L",l,"sym/antisym norms",out)
```