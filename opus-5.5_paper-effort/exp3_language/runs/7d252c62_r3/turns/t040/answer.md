```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,26); B=topright(OV5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:24]]
dirs={}
for j in range(6):
    Q,K,V,O=mats(0,j); M=Q.T@K
    U,S,Vt=np.linalg.svd(M@T); dirs[j]=T@Vt[0]; 
uA=dirs[5]; uB=dirs[3]
print("cos uA,uB", abs(uA@uB))
# how much of OV5 / OV1 copies uA, uB relative to average token dir
for nm,u in [('uA',uA),('uB',uB)]:
    print(nm, "|OV5 u|", np.linalg.norm(OV5@u).round(2), "|OV1 u|", np.linalg.norm(OV1@u).round(2), "avg over T dirs", (np.linalg.norm(OV5@T)/np.sqrt(24)).round(2), (np.linalg.norm(OV1@T)/np.sqrt(24)).round(2))
    F=W['W_fc.0']*W['ln2_w.0']
    print("   MLP0 read |F u|", np.linalg.norm(F@u).round(2), "avg T", (np.linalg.norm(F@T)/np.sqrt(24)).round(2))
    # induction forms: value on u (current token u at query, prev token u at key)
    for (l,j) in [(2,1),(3,1),(3,5),(2,2)]:
        Q,K,V,O=mats(l,j); M=Q.T@K
        G=M@OV5
        val=u@G@u
        rand=np.trace(T.T@G@T)/24
        print(f"   L{l}H{j} u^T G u {val:.3f}  avg diag {rand:.3f}")
# L0H3 query side: which positional dirs, and its keypref in u
for j in (3,2,1,5):
    Q,K,V,O=mats(0,j); s=slice(64*j,64*j+64)
    kp=K.T@(W['W_q.0'][s]@W['ln1_b.0'])/8
    print(j,"keypref . uA", (kp@uA).round(3), "uB", (kp@uB).round(3))
```