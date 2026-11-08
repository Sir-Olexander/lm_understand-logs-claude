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
def ov(l,j): Q,K,V,O=H[l][j]; return O@V
# token-ish subspace: OV-in of H3, H0 combined; positional-ish: QK spaces of H3,H0
U,s,Vt=np.linalg.svd(np.vstack([ov(0,3)/np.linalg.norm(ov(0,3)),ov(0,0)/np.linalg.norm(ov(0,0))])); 
A3=H[0][3][0].T@H[0][3][1]; A0=H[0][0][0].T@H[0][0][1]
Up,sp,_=np.linalg.svd(np.hstack([A3,A3.T,A0,A0.T]))
print("token-sub sv",s[:40]); print("pos-sub sv",sp[:40])
for kt,kp_ in [(18,12),(28,20),(40,30)]:
    T=Vt[:kt].T; P=Up[:,:kp_]
    print("k",kt,kp_,"T-P overlap", np.linalg.norm(T.T@P)**2/min(kt,kp_))
    for j in [1,2,4,5]:
        Q,K,V,O=H[0][j]; A=Q.T@K
        Ua,sa,Vat=np.linalg.svd(A); qd=Ua[:,0]; kd=Vat[0]
        Uv,sv,Vvt=np.linalg.svd(O@V); vin=Vvt[0]; vin2=Vvt[1]
        f=lambda v: (round(float(np.linalg.norm(T.T@v)),2), round(float(np.linalg.norm(P.T@v)),2))
        print("  H",j,"qd T,P",f(qd)," kd",f(kd)," vin1",f(vin)," vin2",f(vin2))
```