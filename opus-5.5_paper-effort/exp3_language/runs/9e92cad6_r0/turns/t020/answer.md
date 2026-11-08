```python
w2=W['ln2_w.0']
Wfc=W['W_fc.0']*w2
Uo,So,_=np.linalg.svd(W['W_o.0'],full_matrices=False)
for r in [0,20,40,60,80]:
    O=Uo[:,:r]
    Pr=np.eye(384)-O@O.T
    s=np.linalg.svd(Wfc@Pr,compute_uv=False)
    print(r,np.round(s[:45],2))
# Also W_v.0 and W_q.0/W_k.0 (layer-0 input) spectra stacked
w1=W['ln1_w.0']
A=np.concatenate([W['W_q.0']*w1,W['W_k.0']*w1,W['W_v.0']*w1],0)
print("stacked qkv.0 sv",np.round(np.linalg.svd(A,compute_uv=False)[:80],2))
```