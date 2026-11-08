```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
w0=W["ln1_w.0"]; b0=W["ln1_b.0"]
def top(M,r):
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return Vt[:r].T, s[:r], U[:,:r]
T,_,_=top((hm("W_v",0,3)*w0),18)
P,_,_=top(np.concatenate([hm("W_q",0,3)*w0,hm("W_k",0,3)*w0],0),16)
def proj(v,S): return np.linalg.norm(S.T@v)/np.linalg.norm(v)
dirs={}
for j in range(6):
    Qv,qs,qU=top(hm("W_q",0,j)*w0,2); Kv,ks,kU=top(hm("W_k",0,j)*w0,2); Vv,vs,_=top(hm("W_v",0,j)*w0,3)
    dirs[j]=(Qv,Kv,Vv)
    qb=hm("W_q",0,j)@b0
    print(f"L0H{j}: Qsv {qs} Ksv {ks} Vsv {vs} |Wq b0| {np.linalg.norm(qb):.3f} |Wk b0| {np.linalg.norm(hm('W_k',0,j)@b0):.3f}")
    for nm,D in [("Q",Qv),("K",Kv),("V",Vv)]:
        print(f"   {nm} dirs: in T {[round(proj(D[:,i],T),2) for i in range(D.shape[1])]}  in P {[round(proj(D[:,i],P),2) for i in range(D.shape[1])]}")
print("cos between K top dirs of heads 1,2,4,5 and V top dirs:")
for a in [1,2,4,5]:
    print(a, [round(abs(dirs[a][1][:,0]@dirs[b][1][:,0]),2) for b in [1,2,4,5]], [round(abs(dirs[a][1][:,0]@dirs[b][2][:,0]),2) for b in [1,2,4,5]],[round(abs(dirs[a][0][:,0]@dirs[b][0][:,0]),2) for b in [1,2,4,5]])
```