```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
hd=64
def heads(l):
    lw=W[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        Wq=W[f'W_q.{l}'][s]*lw; Wk=W[f'W_k.{l}'][s]*lw; Wv=W[f'W_v.{l}'][s]*lw; Wo=W[f'W_o.{l}'][:,s]
        out.append((Wq,Wk,Wv,Wo))
    return out
H={l:heads(l) for l in range(4)}
OV={(l,j):H[l][j][3]@H[l][j][2] for l in range(4) for j in range(6)}
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
U,S,Vt=np.linalg.svd(X,full_matrices=False)
P=Vt[:30].T
def outsp(l,j,k): return np.linalg.svd(OV[(l,j)],full_matrices=False)[0][:,:k]
Pos=np.linalg.qr(np.hstack([P,outsp(0,4,2),outsp(0,0,2)]))[0]
projPos=np.eye(384)-Pos@Pos.T
A_=OV[(0,1)]@projPos; B_=OV[(0,5)]@projPos
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
Prev=np.linalg.qr(np.hstack([outsp(0,5,24),outsp(0,1,24)]))[0]
Prev=Prev-Pos@(Pos.T@Prev); Prev=np.linalg.qr(Prev)[0]
lb={l:W[f'ln1_b.{l}'] for l in range(4)}
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        QK=Wq.T@Wk; tot=np.linalg.norm(QK)
        blocks={}
        for qn,Qb in [('pos',Pos),('tok',T),('prv',Prev)]:
            for kn,Kb in [('pos',Pos),('tok',T),('prv',Prev)]:
                blocks[qn+'>'+kn]=np.linalg.norm(Qb.T@QK@Kb)/tot
        qb=W[f'W_q.{l}'][j*64:(j+1)*64]@lb[l]; kp=Wk.T@qb
        print(f'L{l}H{j} |QK|={tot:5.1f} '+' '.join(f'{k}={v:.2f}' for k,v in blocks.items())+f' | bias>pos={np.linalg.norm(Pos.T@kp):.1f} bias>all={np.linalg.norm(kp):.1f}')
```