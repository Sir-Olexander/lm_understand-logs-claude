```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
W64={k:v.astype(np.float64) for k,v in W.items()}
def heads(l):
    g=W64[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W64[f'W_q.{l}'][s]*g, W64[f'W_k.{l}'][s]*g, W64[f'W_v.{l}'][s]*g, W64[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
for j in [0,3]:
    Q,K,V,O=H[0][j]
    print(j,"OV",np.linalg.svd(O@V,compute_uv=False)[:45])
    print(j,"QK",np.linalg.svd(Q.T@K,compute_uv=False)[:45])
    print(j,"V ",np.linalg.svd(V,compute_uv=False)[:45])
# random baseline for V*g
g=W64['ln1_w.0']
R=np.random.randn(64,384)*0.02*g
print("rand V*g",np.linalg.svd(R,compute_uv=False)[:5])
print("Wfc0*g2", np.linalg.svd(W64['W_fc.0']*W64['ln2_w.0'],compute_uv=False)[:60])
```