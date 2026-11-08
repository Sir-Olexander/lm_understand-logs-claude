```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
def topleft(M,k):
    U,S,Vt=np.linalg.svd(M); return U[:,:k]
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
A=topright(O1@V1,26); B=topright(O5@V5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:20]]
Qs=[]
for j in (1,5):
    Q,K,V,O=mats(0,j); Qs.append(topright(Q,30)); Qs.append(topright(K,30))
Pp=sum(x@x.T for x in Qs)/4
ev2,evec2=np.linalg.eigh(Pp); idx2=np.argsort(-ev2); Ppos=evec2[:,idx2[:24]]
Ppos=np.linalg.qr(Ppos-T@(T.T@Ppos))[0]
subs={'T':T,'pos':Ppos}
for j in range(6):
    Q,K,V,O=mats(0,j)
    subs[f'o0.{j}']=np.linalg.qr(O@V@T)[0] if j in (1,5) else topleft(O@V,3)
U0,S0,_=np.linalg.svd(W['W_proj.0']); subs['mlp0top8']=U0[:,:8]
print("W_proj.0 sv", np.round(S0[:12],2))
for nm in subs: print(nm, subs[nm].shape[1], end='; ')
print()
for l in (1,2):
  for j in range(6):
    Q,K,V,O=mats(l,j)
    U,S,Vt=np.linalg.svd(Q.T@K)
    for r in range(2):
        print(f"L{l}H{j} r{r} s{S[r]:.2f} q:", " ".join(f"{nm}:{np.linalg.norm(X.T@U[:,r]):.2f}" for nm,X in subs.items()), "| k:", " ".join(f"{nm}:{np.linalg.norm(X.T@Vt[r]):.2f}" for nm,X in subs.items()))
```