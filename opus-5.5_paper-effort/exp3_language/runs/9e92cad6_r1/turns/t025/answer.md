```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
w0=W["ln1_w.0"]
Tr=np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0)
L1r=np.concatenate([hm(nm,1,j)*W["ln1_w.1"] for nm in ["W_q","W_k","W_v"] for j in range(6)],0)
def run(ks):
    cols=[]
    for j,k in zip([3,0,1,2,4,5],ks):
        if k>0:
            U,s,_=np.linalg.svd(OV(0,j)); cols.append(U[:,:k])
    E=np.zeros((384,len(out))); E[out,range(len(out))]=1; cols.append(E)
    A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
    FC=(W["W_fc.0"]*W["ln2_w.0"])@Proj
    _,s,Vt=np.linalg.svd(FC,full_matrices=False)
    def frac(M,v): M=M/np.linalg.norm(M); return (np.linalg.norm(M@v)**2)*384
    eT=[frac(Tr,Vt[k]) for k in range(32)]; eL=[frac(L1r,Vt[k]) for k in range(32)]
    print("ks",ks); print("  sv  ", s[20:32]); print("  T-enr", np.array(eT[20:32])); print("  L1-enr", np.array(eL[20:32]))
run([18,10,2,2,2,3]); run([40,30,4,4,4,4]); run([0,0,0,0,0,0])
```