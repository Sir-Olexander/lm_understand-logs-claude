```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
def lsub(M,k): U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
cols=[lsub(OV(0,j),k) for j,k in zip([3,0,1,2,4,5],[25,15,3,3,3,3])]
E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
_,_,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False); T=Vt[:26].T
w=W["ln1_w.3"]; Q=hm("W_q",3,3)*w; K=hm("W_k",3,3)*w
forms={"ind03":T.T@Q.T@K@OV(0,3)@T}
for (l,j) in [(1,0),(1,5),(1,1),(2,1),(2,5),(2,3)]:
    forms[f"ind{l}{j}"]=T.T@Q.T@K@OV(l,j)@T
forms["same"]=T.T@Q.T@K@T
for n,M in forms.items():
    d=np.diag(M); print(f"{n}: trace {np.trace(M):6.2f}  |M| {np.linalg.norm(M):5.2f}  sym-part posdef frac {np.sum(np.linalg.eigvalsh((M+M.T)/2)>0)}/26")
M1=forms["ind03"]
for n2 in ["ind10","ind15","ind21","ind25","same"]:
    M2=forms[n2]
    ev,EV=np.linalg.eig(np.linalg.solve(M1,M2))
    # test diagonalization of a third form with these eigvecs
    Einv=np.linalg.inv(EV)
    for n3 in ["ind11","ind23"]:
        D=EV.T@forms[n3]@EV
        off=np.linalg.norm(D-np.diag(np.diag(D)))/np.linalg.norm(D)
        # random baseline: random invertible basis
        R=np.random.randn(26,26); Dr=R.T@forms[n3]@R; offr=np.linalg.norm(Dr-np.diag(np.diag(Dr)))/np.linalg.norm(Dr)
        print(f"  basis from ind03^-1 {n2}: imag frac {np.abs(ev.imag).sum()/np.abs(ev).sum():.2f}; {n3} offdiag {off:.2f} (random {offr:.2f})")
```