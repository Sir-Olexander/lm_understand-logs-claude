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
P=Vt[:20].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
vecs={}; names=[]
for l in range(0,4):
    lb=W[f'ln1_b.{l}']
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        qb=W[f'W_q.{l}'][j*64:(j+1)*64]@lb
        kp=Wk.T@qb
        vecs[f'L{l}H{j}tok']=T.T@kp
        vecs[f'L{l}H{j}prv']=T.T@(OV[(0,5)].T@kp)
# L0 token-key heads: top key-token dir of QK (complement P)
for j in [0,2,3]:
    Wq,Wk,_,_=H[0][j]
    u,s,vt=np.linalg.svd((Wq.T@Wk)@projP); vecs[f'X0H{j}']=T.T@vt[0]
names=[n for n in vecs if np.linalg.norm(vecs[n])>0]
sel=[n for n in names if n.startswith('X') or (n[1] in '12' and (n.endswith('tok') or n.endswith('prv')))]
V=np.array([vecs[n]/np.linalg.norm(vecs[n]) for n in sel])
C=V@V.T
print('norms:',{n:round(float(np.linalg.norm(vecs[n])),2) for n in sel})
print('      '+' '.join(f'{n[:7]:>7}' for n in sel))
for i,n in enumerate(sel):
    print(f'{n:>10} '+' '.join(f'{C[i,k]:7.2f}' for k in range(len(sel))))
```