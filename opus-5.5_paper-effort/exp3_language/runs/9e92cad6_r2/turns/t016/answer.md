```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
for l in range(4):
    lw=W[f'ln1_w.{l}']; lb=W[f'ln1_b.{l}']
    row=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        Wq=W[f'W_q.{l}'][s]; Wk=W[f'W_k.{l}'][s]; Wv=W[f'W_v.{l}'][s]
        qb=np.linalg.norm(Wq@lb); qc=np.linalg.norm(Wq*lw)
        kb=np.linalg.norm(Wk@lb); kc=np.linalg.norm(Wk*lw)
        vb=np.linalg.norm(Wv@lb); vc=np.linalg.norm(Wv*lw)
        row.append('H%d q %.2f/%.2f k %.2f/%.2f v %.2f/%.2f'%(j,qb,qc,kb,kc,vb,vc))
    print('L%d |lb| %.2f |lw| %.2f'%(l,np.linalg.norm(lb),np.linalg.norm(lw)), ' | '.join(row))
```