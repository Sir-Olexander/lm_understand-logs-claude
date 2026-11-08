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
def lsv(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k]
Tb=rsv(np.vstack([OV[(0,1)],OV[(0,5)]]),24)
rng=np.random.default_rng(1)
def randorth(k):
    return np.linalg.qr(rng.normal(size=(384,k)))[0]
for l in range(1,4):
    for j in range(6):
        Wq,Wk,Wv,Wo=H[l][j]
        out=[]
        for src in [(0,1),(0,5),(0,2),(0,3)]:
            M=Tb.T@Wq.T@Wk@OV[src]@Tb
            sv=np.linalg.svd(M,compute_uv=False)
            ev=np.linalg.eigvals(M)
            out.append(f'{src[1]}: |M|={np.linalg.norm(M):.2f} tr/nuc={np.trace(M)/sv.sum():+.2f}')
        # also direct token-token QK
        Md=Tb.T@Wq.T@Wk@Tb; svd=np.linalg.svd(Md,compute_uv=False)
        print(f'L{l}H{j} '+' | '.join(out)+f' || direct tok-tok |M|={np.linalg.norm(Md):.2f} tr/nuc={np.trace(Md)/svd.sum():+.2f}')
    print()
# how much do queries read Tb vs random 24-d
for l in range(1,4):
    R=randorth(24)
    print('layer',l,'q-read frac Tb vs random:',[round(np.linalg.norm(H[l][j][0]@Tb)**2/np.linalg.norm(H[l][j][0])**2,3) for j in range(6)], round(np.linalg.norm(H[l][0][0]@R)**2/np.linalg.norm(H[l][0][0])**2,3))
    print('        k-read frac Tb :',[round(np.linalg.norm(H[l][j][1]@Tb)**2/np.linalg.norm(H[l][j][1])**2,3) for j in range(6)])
    O=lsv(np.hstack([OV[(0,1)],OV[(0,5)]]),24)
    print('        k-read frac out(L0H1,H5) :',[round(np.linalg.norm(H[l][j][1]@O)**2/np.linalg.norm(H[l][j][1])**2,3) for j in range(6)])
    print('        q-read frac out(L0H1,H5) :',[round(np.linalg.norm(H[l][j][0]@O)**2/np.linalg.norm(H[l][j][0])**2,3) for j in range(6)])
    print('        v-read frac out(L0H1,H5) :',[round(np.linalg.norm(H[l][j][2]@O)**2/np.linalg.norm(H[l][j][2])**2,3) for j in range(6)])
    print('        v-read frac Tb :',[round(np.linalg.norm(H[l][j][2]@Tb)**2/np.linalg.norm(H[l][j][2])**2,3) for j in range(6)])
```