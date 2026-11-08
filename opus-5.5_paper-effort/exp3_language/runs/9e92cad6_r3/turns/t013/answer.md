```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
W64={k:v.astype(np.float64) for k,v in W.items()}
def heads(l):
    g=W64[f'ln1_w.{l}']
    out=[]
    for j in range(6):
        s=slice(64*j,64*j+64)
        out.append((W64[f'W_q.{l}'][s]*g, W64[f'W_k.{l}'][s]*g, W64[f'W_v.{l}'][s]*g, W64[f'W_o.{l}'][:,s]))
    return out
H=[heads(l) for l in range(4)]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M)
    return U[:,:k] if side=='L' else Vt[:k].T
def ov(l,j): Q,K,V,O=H[l][j]; return O@V
subs={}
subs['T']=sub(ov(0,3),18,'R')
subs['P']=sub(H[0][3][0].T@H[0][3][1]+ (H[0][3][0].T@H[0][3][1]).T,12,'L')
subs['Fout']=sub(np.hstack([ov(0,j) for j in [1,2,4,5]]),2,'L')
subs['H3o']=sub(ov(0,3),18,'L')
subs['H0o']=sub(ov(0,0),8,'L')
names=list(subs)
for n in names: 
    print(n, [round(float(np.linalg.norm(subs[n].T@subs[m])**2/min(subs[n].shape[1],subs[m].shape[1])),2) for m in names])
for l in range(1,4):
    for j in range(6):
        Q,K,V,O=H[l][j]; A=Q.T@K; tot=np.linalg.norm(A)
        qfr=[np.linalg.norm(subs[n].T@A)/tot for n in names]
        kfr=[np.linalg.norm(A@subs[n])/tot for n in names]
        vfr=[np.linalg.norm(V@subs[n])/np.linalg.norm(V) for n in names]
        print(f"L{l}H{j} q:",np.array(qfr)," k:",np.array(kfr)," v:",np.array(vfr))
print(names)
```