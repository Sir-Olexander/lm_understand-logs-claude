```python
np.set_printoptions(precision=3, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
Q2,K2,V2,O2=mats(0,2)
U2,s2,Vt2=np.linalg.svd(O2@V2); ins=Vt2[:2].T; outs=U2[:,:2]
for j in range(6):
    Q,K,V,O=mats(0,j); QK=Q.T@K/8
    print(j, 'ins^T QK ins *384:\n', ins.T@QK@ins*384)
# also: fraction of each L0 head's Q,K,V energy in ins
for j in range(6):
    Q,K,V,O=mats(0,j)
    f=lambda M: np.linalg.norm(M@ins)**2/np.linalg.norm(M)**2*192
    print(j,'Q %.1f K %.1f V %.1f'%(f(Q),f(K),f(V)))
# MLP0 read of ins and outs
Wfc=W['W_fc.0']*W['ln2_w.0']
print('fc0 read ins x%.1f outs x%.1f'%(np.linalg.norm(Wfc@ins)**2/np.linalg.norm(Wfc)**2*192, np.linalg.norm(Wfc@outs)**2/np.linalg.norm(Wfc)**2*192))
for l in range(1,4):
    Wfc=W[f'W_fc.{l}']*W[f'ln2_w.{l}']
    print('fc',l,'read ins x%.1f outs x%.1f'%(np.linalg.norm(Wfc@ins)**2/np.linalg.norm(Wfc)**2*192, np.linalg.norm(Wfc@outs)**2/np.linalg.norm(Wfc)**2*192))
    for j in range(6):
        Q,K,V,O=mats(l,j)
        f=lambda M,S: np.linalg.norm(M@S)**2/np.linalg.norm(M)**2*192
        print('  L%dH%d ins: Q %.1f K %.1f V %.1f | outs: Q %.1f K %.1f V %.1f'%(l,j,f(Q,ins),f(K,ins),f(V,ins),f(Q,outs),f(K,outs),f(V,outs)))
```