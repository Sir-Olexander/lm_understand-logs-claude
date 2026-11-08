```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
out=[21,39,58,99,184]
def lsub(M,k): U,s,_=np.linalg.svd(M,full_matrices=False); return U[:,:k]
w0=W["ln1_w.0"]
def frac(M,v): M=M/np.linalg.norm(M); return (np.linalg.norm(M@v)**2)*384
readers={"V0V3":np.concatenate([hm("W_v",0,3)*w0,hm("W_v",0,0)*w0],0),
 "L1qk":np.concatenate([hm(n,1,j)*W["ln1_w.1"] for n in ["W_q","W_k"] for j in range(6)],0),
 "L1v":np.concatenate([hm("W_v",1,j)*W["ln1_w.1"] for j in range(6)],0),
 "L0K124":np.concatenate([hm("W_k",0,j)*w0 for j in [1,2,4]],0),
 "L3H3q":hm("W_q",3,3)*W["ln1_w.3"]}
for ks in [[12,6,2,2,2,3],[25,15,3,3,3,3]]:
    cols=[lsub(OV(0,j),k) for j,k in zip([3,0,1,2,4,5],ks)]
    E=np.zeros((384,5)); E[out,range(5)]=1; cols.append(E)
    A=np.linalg.qr(np.concatenate(cols,1))[0]; Proj=np.eye(384)-A@A.T
    _,s,Vt=np.linalg.svd((W["W_fc.0"]*W["ln2_w.0"])@Proj,full_matrices=False)
    print("ks",ks,"sv", s[18:32])
    for n,R in readers.items():
        print(f"  {n:7s}", np.array([frac(R,Vt[k]) for k in range(18,32)]))
```