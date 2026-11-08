```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def hm(nm,l,j):
    M=W[f"{nm}.{l}"]
    return M[64*j:64*j+64] if nm!="W_o" else M[:,64*j:64*j+64]
out=[21,39,58,99,184]
keep=np.ones(384,bool); keep[out]=False
def orth(M,r):
    M=M.copy(); M[~keep]=0
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:r]
w0=W["ln1_w.0"]
def OV(l,j): return hm("W_o",l,j)@(hm("W_v",l,j)*W[f"ln1_w.{l}"])
S={}
S['T(V3in)']=orth((hm("W_v",0,3)*w0).T,18)
S['P(QK3in)']=orth(np.concatenate([hm("W_q",0,3)*w0,hm("W_k",0,3)*w0],0).T,16)
S['O3']=orth(OV(0,3),18)
S['O0']=orth(OV(0,0),8)
S['O5']=orth(OV(0,5),3)
S['MLP0']=orth(W['W_proj.0'],40)
for l in range(1,3):
    for j in range(6): S[f'O{l}{j}']=orth(OV(l,j),12)
names=list(S.keys())
print("subspace overlaps T vs P", np.linalg.svd(S['T(V3in)'].T@S['P(QK3in)'],compute_uv=False)[:6])
print("O3 vs T", np.linalg.svd(S['O3'].T@S['T(V3in)'],compute_uv=False)[:6], " O3 vs P", np.linalg.svd(S['O3'].T@S['P(QK3in)'],compute_uv=False)[:6])
print("enrichment (fraction of eff weight energy in subspace / (r/379))")
print("        "+" ".join(f"{n[:7]:>7}" for n in names))
for l in range(1,4):
    w=W[f"ln1_w.{l}"]
    for j in range(6):
        for nm in ["W_q","W_k","W_v"]:
            M=hm(nm,l,j)*w; M[:,~keep]=0
            tot=np.linalg.norm(M)**2
            row=[]
            for n in names:
                if n.startswith('O') and len(n)==3 and int(n[1])>=l: row.append("      -"); continue
                U=S[n]; f=np.linalg.norm(M@U)**2/tot/(U.shape[1]/379); row.append(f"{f:7.2f}")
            print(f"L{l}H{j}{nm[-1]} "+" ".join(row))
```