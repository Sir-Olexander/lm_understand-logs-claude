```python
import numpy as np
np.set_printoptions(linewidth=180, precision=2, suppress=True)
def head(l,h):
    s=slice(64*h,64*h+64); w=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def cos(A,B): return np.linalg.svd(A.T@B,compute_uv=False)
for h in range(6):
    q,k,v,o=head(0,h)
    sA=np.linalg.svd(q.T@k,compute_uv=False); sO=np.linalg.svd(o@v,compute_uv=False); sV=np.linalg.svd(v,compute_uv=False)
    print(f"L0H{h} QK sv:",sA[:12]); print("      OV sv:",sO[:30]); print("      V sv :",sV[:30])
# subspaces
subs={}
for h in range(6):
    q,k,v,o=head(0,h)
    U,s,Vt=np.linalg.svd(q.T@k); subs[f'Q{h}']=U[:,:6]; subs[f'K{h}']=Vt[:6].T
    U,s,Vt=np.linalg.svd(o@v); subs[f'Vin{h}']=Vt[:20].T
print("overlap of OV-input spaces (20d) between L0 heads (mean cos^2):")
for a in range(6):
    print(a,[round(float((cos(subs[f'Vin{a}'],subs[f'Vin{b}'])**2).mean()),2) for b in range(6)])
print("random 20d vs 20d mean cos^2 ~", round(20/384,3))
for h in [2,3,0,4]:
    for nm in ['Q','K']:
        S=subs[f'{nm}{h}']
        print(f"L0H{h} {nm}-side top6 dirs: cos^2 with Vin0,Vin1,Vin5 union:",
              np.round([(np.linalg.norm(np.linalg.qr(np.hstack([subs['Vin0'],subs['Vin1'],subs['Vin5']]))[0].T@S[:,i])**2) for i in range(6)],2))
```