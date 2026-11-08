```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
def sub(M,k,side):
    U,s,Vt=np.linalg.svd(M); return U[:,:k] if side=='out' else Vt[:k].T
Q3,K3,V3,O3=mats(0,3); T=sub(O3@V3,25,'in'); 
Fc=W['W_fc.0']*W['ln2_w.0']
G=sum(mats(0,j)[3]@mats(0,j)[2] for j in [1,2,4,5]); outs=sub(G,3,'out')
def Pset(k,kt):
    cols=[]
    for j in range(6):
        Q,K,V,O=mats(0,j); cols+= [sub(Q,k,'in'), sub(K,k,'in')]
    A=np.hstack(cols); Tk=sub(O3@V3,kt,'in'); A=A-Tk@(Tk.T@A); return np.linalg.svd(A,full_matrices=False)[0]
for (k,kt,np_,ns3,ns0) in [(10,25,30,25,10),(10,25,20,25,10),(10,25,40,25,10),(5,20,20,20,8),(10,25,30,0,0),(10,25,0,25,10),(10,25,0,0,0)]:
    blocks=[outs]
    if np_>0: blocks.append(Pset(k,kt)[:,:np_])
    if ns3>0: blocks.append(sub(O3@V3,ns3,'out'))
    if ns0>0: blocks.append(sub(mats(0,0)[3]@mats(0,0)[2],ns0,'out'))
    X=np.linalg.qr(np.hstack(blocks))[0]
    Fr=Fc-(Fc@X)@X.T; s=np.linalg.svd(Fr,compute_uv=False)
    r=s[:40]/s[1:41]
    top=np.argsort(-r[5:38])[:3]+5
    print((k,kt,np_,ns3,ns0),'largest ratio gaps after index',top+1,np.round(r[top],3), 'svals 20-32',np.round(s[20:32],2))
```