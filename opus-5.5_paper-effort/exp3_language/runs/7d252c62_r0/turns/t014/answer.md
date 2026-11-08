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
P=Vt[:30].T; proj=np.eye(384)-P@P.T
A_=OV[(0,1)]@proj; B_=OV[(0,5)]@proj
U2,S2,Vt2=np.linalg.svd(np.vstack([A_/np.linalg.norm(A_),B_/np.linalg.norm(B_)]),full_matrices=False)
def forms(T):
    F={}
    for j in range(6):
        Wq,Wk,_,_=H[1][j]; F[f'A1{j}']=T.T@Wq.T@Wk@T
    for (l,j) in [(2,1),(3,5),(3,1),(2,2),(3,0)]:
        Wq,Wk,_,_=H[l][j]; F[f'B{l}{j}']=T.T@Wq.T@Wk@OV[(0,5)]@T
        F[f'C{l}{j}']=T.T@OV[(0,5)].T@Wq.T@Wk@OV[(0,1)]@T
    return F
def geig(A,B):
    ev,V=np.linalg.eig(np.linalg.solve(B,A))
    V=V/np.linalg.norm(V,axis=0)
    return ev,V
for kT in [24,27,28]:
    T=Vt2[:kT].T
    F=forms(T)
    pairs=[('A11','B21'),('A13','B35'),('A15','B31'),('A12','B35'),('A14','B21')]
    res={}
    for a,b in pairs:
        ev,V=geig(F[a],F[b]); res[(a,b)]=(ev,V)
        print(kT,a,b,'frac real ev',np.mean(np.abs(ev.imag)<1e-6*np.abs(ev).max()+1e-9).round(2), 'ev',np.sort_complex(ev)[:6].round(2))
    keys=list(res)
    for i in range(len(keys)):
        for k in range(i+1,len(keys)):
            V1=np.real(res[keys[i]][1]); V2=np.real(res[keys[k]][1])
            V1=V1/np.linalg.norm(V1,axis=0); V2=V2/np.linalg.norm(V2,axis=0)
            C=np.abs(V1.T@V2)
            print('   match',keys[i],keys[k],'best cos per vec',np.sort(C.max(1))[::-1].round(2))
```