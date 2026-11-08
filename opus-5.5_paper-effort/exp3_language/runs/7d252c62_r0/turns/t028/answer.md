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
def lsv(M,k): return np.linalg.svd(M,full_matrices=False)[0][:,:k]
def rsv(M,k): return np.linalg.svd(M,full_matrices=False)[2][:k].T
writers={}
for l in range(4):
    for j in range(6): writers[f'A{l}{j}']=lsv(OV[(l,j)],8)
    writers[f'M{l}']=lsv(W[f'W_proj.{l}'],32)
for (l,j) in [(3,3),(0,4),(2,5),(3,2),(3,4)]:
    Wq,Wk,Wv,Wo=H[l][j]
    M=Wo@Wv
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    print(f'L{l}H{j} OV sv',S[:8].round(2))
    for r in range(3):
        v=Vt[r]; u=U[:,r]
        rd={k:np.linalg.norm(B.T@v) for k,B in writers.items() if int(k[1])<l or (k[0]=='M' and int(k[1])<l)}
        top=sorted(rd.items(),key=lambda x:-x[1])[:6]
        print(f'   in-dir {r}: reads from',[(k,round(float(x),2)) for k,x in top])
        # does output direction feed later components?
        if l<3:
            rd2={}
            for l2 in range(l+1,4):
                lw=W[f'ln1_w.{l2}']
                for j2 in range(6):
                    for nm,Mx in [('q',H[l2][j2][0]),('k',H[l2][j2][1]),('v',H[l2][j2][2])]:
                        rd2[f'L{l2}H{j2}{nm}']=np.linalg.norm(Mx@u)/np.linalg.norm(Mx)*np.sqrt(384)
            top2=sorted(rd2.items(),key=lambda x:-x[1])[:6]
            print(f'   out-dir {r} read by',[(k,round(float(x),2)) for k,x in top2])
```