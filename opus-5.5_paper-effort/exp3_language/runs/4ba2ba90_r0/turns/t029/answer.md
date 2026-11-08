```python
for d_ in [222,78,360,72,103,254,25,197]:
    print(d_, [round(float(W[f'{n}.{l}'][d_]),3) for l in range(4) for n in ['ln1_w','ln2_w']], [round(float(W[f'{n}.{l}'][d_]),3) for l in range(4) for n in ['ln1_b','ln2_b']])
# what writes into dim 222 overall: row norms of W_o.l and W_proj.l at 222 vs typical
for l in range(4):
    for nm in ['W_o','W_proj']:
        M=W[f'{nm}.{l}']; rn=np.linalg.norm(M,axis=1)
        print(l,nm,'dim222',round(float(rn[222]),2),'dim78',round(float(rn[78]),2),'median',round(float(np.median(rn)),2),'max dim',int(np.argmax(rn)),round(float(rn.max()),2))
```