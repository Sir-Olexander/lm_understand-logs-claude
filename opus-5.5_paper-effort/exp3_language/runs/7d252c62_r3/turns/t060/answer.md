```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
U4,S4,_=np.linalg.svd(mats(0,4)[3]@mats(0,4)[2]); o4=U4[:,:2]
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=mats(l,j); M=Q.T@K/8
        blk=o4.T@M@o4
        ev=np.linalg.eigvals(blk)
        b=W[f'ln1_b.{l}']; s=slice(64*j,64*j+64)
        kp=(K.T@(W[f'W_q.{l}'][s]@b))/8
        kpf=o4.T@kp
        # fraction of QK energy with key in f-plane
        fk=np.linalg.norm(M@o4)**2/np.linalg.norm(M)**2
        print(f"L{l}H{j} f-plane block {np.round(blk,2).tolist()} eig {np.round(np.abs(ev),2)}@{np.round(np.degrees(np.angle(ev)),0)} | keypref f-plane {np.round(kpf,2)} |QK key in f-plane frac {fk:.2f}")
```