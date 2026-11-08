```python
w2=W['ln2_w.0']; w1=W['ln1_w.0']
Wfc=W['W_fc.0']*w2
Uo,So,_=np.linalg.svd(W['W_o.0'],full_matrices=False)
O=Uo[:,:60]; Pr=np.eye(384)-O@O.T
U,S,Vt=np.linalg.svd(Wfc@Pr,full_matrices=False)
Tm=Vt[:26].T
# positional subspace from L0H3 QK (in ln1-folded coordinates!) -> note: folded by ln1_w; convert to raw residual coords
def heads(l):
    w=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s=slice(j*64,(j+1)*64)
        q=W[f'W_q.{l}'][s]*w; k=W[f'W_k.{l}'][s]*w; v=W[f'W_v.{l}'][s]*w; o=W[f'W_o.{l}'][:,s]
        res.append((q,k,v,o))
    return res
H=[heads(l) for l in range(4)]
q,k,v,o=H[0][3]; Uq,Sq,Vq=np.linalg.svd(q.T@k)
P=np.linalg.svd(np.concatenate([Uq[:,:16],Vq[:16].T],1),full_matrices=False)[0][:,:20]
T3=np.linalg.svd(o@v)[2][:16].T
print("Tm(26) vs P cos:",np.round(np.linalg.svd(Tm.T@P,compute_uv=False),2))
print("Tm(26) vs T3 cos:",np.round(np.linalg.svd(Tm.T@T3,compute_uv=False),2))
print("Tm(26) vs O(60) cos:",np.round(np.linalg.svd(Tm.T@O,compute_uv=False)[:5],2))
# what fraction of L0H3 QK lives in Tm?
QK=q.T@k
print("L0H3 QK in Tm x Tm:",np.linalg.norm(Tm.T@QK@Tm)/np.linalg.norm(QK), " in P x P:",np.linalg.norm(P.T@QK@P)/np.linalg.norm(QK))
# how much of L0H3 OV input lives in Tm
OV=o@v
print("L0H3 OV input energy in Tm:",np.linalg.norm(OV@Tm)**2/np.linalg.norm(OV)**2, "in P:",np.linalg.norm(OV@P)**2/np.linalg.norm(OV)**2)
for j in range(6):
    q,k,v,o=H[0][j]; OV=o@v; QK=q.T@k
    print(j,"OV-in in Tm",round(np.linalg.norm(OV@Tm)**2/np.linalg.norm(OV)**2,2),"OV-in in P",round(np.linalg.norm(OV@P)**2/np.linalg.norm(OV)**2,2),
          "Q in Tm",round(np.linalg.norm(q@Tm)**2/np.linalg.norm(q)**2,2),"K in Tm",round(np.linalg.norm(k@Tm)**2/np.linalg.norm(k)**2,2),
          "Q in P",round(np.linalg.norm(q@P)**2/np.linalg.norm(q)**2,2),"K in P",round(np.linalg.norm(k@P)**2/np.linalg.norm(k)**2,2))
```