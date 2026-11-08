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
def rsv(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:k].T
Tb=rsv(np.vstack([OV[(0,1)]/np.linalg.norm(OV[(0,1)]),OV[(0,5)]/np.linalg.norm(OV[(0,5)])]),28)
# positional: stacked Q,K of H1,H5, remove Tb
X=np.vstack([H[0][1][0],H[0][1][1],H[0][5][0],H[0][5][1]])
X=X-X@Tb@Tb.T
U,S,Vt=np.linalg.svd(X,full_matrices=False)
print('pos-candidate sv',S[:60].round(2))
P=Vt[:40].T
R=np.linalg.qr(np.random.default_rng(0).normal(size=(384,40)))[0]
for j in range(6):
    Wq,Wk,Wv,Wo=H[0][j]
    QK=Wq.T@Wk
    tot=np.linalg.norm(QK)
    def blk(A,B): return np.linalg.norm(A.T@QK@B)/tot
    print(f'L0H{j} |QK|={tot:.2f} tt={blk(Tb,Tb):.2f} tp={blk(Tb,P):.2f} pt={blk(P,Tb):.2f} pp={blk(P,P):.2f} (rand40x40 {blk(R,R):.2f})  q-bias row k: |b^T QK|')
    Mtt=Tb.T@QK@Tb; sv=np.linalg.svd(Mtt,compute_uv=False); print('      tt tr/nuc %.2f'%(np.trace(Mtt)/sv.sum()))
    Mpp=P.T@QK@P; ev=np.linalg.eigvals(Mpp); idx=np.argsort(-abs(ev))[:12]
    print('      pp ev', ' '.join(f'{abs(e):.2f}@{np.degrees(np.angle(e)):.0f}' for e in ev[idx]))
```