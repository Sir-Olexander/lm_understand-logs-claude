```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); ev=ev[idx]; evec=evec[:,idx]
print(np.round(ev[:40],2))
for dim in [16,20,24]:
    T=evec[:,:dim]; Pt=T@T.T; Pn=np.eye(384)-Pt
    print("dim",dim)
    for j in range(6):
        Q,K,V,O=mats(0,j)
        M=Q.T@K
        tot=np.linalg.norm(M)**2
        parts={ 'tt':np.linalg.norm(Pt@M@Pt)**2/tot,'tn':np.linalg.norm(Pt@M@Pn)**2/tot,'nt':np.linalg.norm(Pn@M@Pt)**2/tot,'nn':np.linalg.norm(Pn@M@Pn)**2/tot}
        ov=O@V; tov=np.linalg.norm(ov@Pt)**2/np.linalg.norm(ov)**2
        print(f" H{j} QK q-tok/k-tok {parts['tt']:.2f} q-tok/k-oth {parts['tn']:.2f} q-oth/k-tok {parts['nt']:.2f} q-oth/k-oth {parts['nn']:.2f} | OV from tok {tov:.2f}")
```