```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
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
proj=np.eye(384)-P@P.T
A=OV[(0,1)]@proj; B=OV[(0,5)]@proj
U2,S2,Vt2=np.linalg.svd(np.vstack([A/np.linalg.norm(A),B/np.linalg.norm(B)]),full_matrices=False)
T=Vt2[:27].T
lb=W['ln1_b.0']
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    QK=Wq.T@Wk
    pt=P.T@QK@T; tt=T.T@QK@T; tp=T.T@QK@P
    print(f'L0H{j} pt sv',np.linalg.svd(pt,compute_uv=False)[:8].round(2),' tt sv',np.linalg.svd(tt,compute_uv=False)[:8].round(2),' tp sv',np.linalg.svd(tp,compute_uv=False)[:6].round(2))
    qb=Wq@lb  # query bias in head space (64)
    kt=(Wk@T).T@qb
    print('     bias->key-token pref norm %.3f'%np.linalg.norm(kt), ' bias->key-pos pref norm %.3f'%np.linalg.norm((Wk@P).T@qb))
    # value: what does OV write, read from T vs P
    print('     OV read: T %.2f  P %.2f'%(np.linalg.norm(OV[(0,j)]@T)**2/np.linalg.norm(OV[(0,j)])**2, np.linalg.norm(OV[(0,j)]@P)**2/np.linalg.norm(OV[(0,j)])**2))
# key-token preference directions of pt top singular vectors: compare across heads
dirs={}
for j in [0,2,3]:
    Wq,Wk,Wv,Wo=H[0][j]
    pt=P.T@(Wq.T@Wk)@T
    u,s,vt=np.linalg.svd(pt); dirs[j]=vt[0]
print('cos between top key-token dirs H0,H2,H3:',abs(dirs[0]@dirs[2]),abs(dirs[0]@dirs[3]),abs(dirs[2]@dirs[3]))
```