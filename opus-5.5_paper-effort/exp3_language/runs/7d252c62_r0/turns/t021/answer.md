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
P=Vt[:30].T; projP=np.eye(384)-P@P.T
A_=OV[(0,1)]@projP; B_=OV[(0,5)]@projP
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
T=Vt2[:27].T
Wfc=W['W_fc.0']*W['ln2_w.0']
a=Wfc@T                 # current token pref
b=Wfc@OV[(0,5)]@T       # prev token pref (via L0H5)
c2=Wfc@OV[(0,1)]@T      # 2-back pref
na=np.linalg.norm(a,axis=1); nb=np.linalg.norm(b,axis=1); nc=np.linalg.norm(c2,axis=1)
print('norm pctiles a',np.percentile(na,[10,50,90]).round(2),' b',np.percentile(nb,[10,50,90]).round(2),' c',np.percentile(nc,[10,50,90]).round(2))
cos=(a*b).sum(1)/(na*nb)
rng=np.random.default_rng(0); perm=rng.permutation(len(a))
cosr=(a*b[perm]).sum(1)/(na*nb[perm])
print('cos(a,b) pctiles',np.percentile(cos,[1,5,25,50,75,95,99]).round(2))
print('random pair    ',np.percentile(cosr,[1,5,25,50,75,95,99]).round(2))
# strongest neurons both a and b large
score=na*nb
top=np.argsort(-score)[:40]
print('top neurons by na*nb: cos',cos[top].round(2))
print('   na',na[top].round(2)); print('   nb',nb[top].round(2))
# among highly aligned neurons (cos>0.6), cluster their a directions
sel=np.where((cos>0.6)&(na>np.percentile(na,50)))[0]
print('n aligned',len(sel))
Z=a[sel]/na[sel,None]; G=Z@Z.T
print('mean pairwise cos among aligned a-dirs', (G.sum()-len(sel))/(len(sel)**2-len(sel)))
u,s,vt=np.linalg.svd(Z,full_matrices=False); print('sv of aligned a-dirs',s[:10].round(2))
```