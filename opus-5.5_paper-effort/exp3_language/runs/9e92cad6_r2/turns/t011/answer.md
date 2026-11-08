```python
np.set_printoptions(precision=2, suppress=True, linewidth=220)
def mats(l,j):
    lw=W[f'ln1_w.{l}']; s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*lw, W[f'W_k.{l}'][s]*lw, W[f'W_v.{l}'][s]*lw, W[f'W_o.{l}'][:,s]
Q3,K3,V3,O3=mats(0,3)
U,s,Vt=np.linalg.svd(O3@V3); Tok=Vt[:15].T   # token-ish input dirs (L0H3 values)
print('L0H3 OV svals',s[:20])
_,sq,Vq=np.linalg.svd(Q3); _,sk,Vk=np.linalg.svd(K3)
print('L0H3 Q sv',sq[:20]); print('L0H3 K sv',sk[:20])
Pq=Vq[:10].T; Pk=Vk[:10].T
Q2,K2,V2,O2=mats(0,2)
U2,s2,Vt2=np.linalg.svd(O2@V2); ins=Vt2[:2].T
print('ins vs Tok', np.linalg.svd(ins.T@Tok,compute_uv=False), 'ins vs PqL0H3',np.linalg.svd(ins.T@Pq,compute_uv=False),'ins vs PkL0H3',np.linalg.svd(ins.T@Pk,compute_uv=False))
print('Tok vs Pq', np.round(np.linalg.svd(Tok.T@Pq,compute_uv=False),2))
print('Tok vs Pk', np.round(np.linalg.svd(Tok.T@Pk,compute_uv=False),2))
for j in [1,2,4,5,0]:
    QK=mats(0,j)[0].T@mats(0,j)[1]/8
    Uq,sv,Vk_=np.linalg.svd(QK)
    for r in range(2):
        u=Uq[:,r]; v=Vk_[r]
        print(j,r,'sv %.3f'%sv[r],' u: ins %.2f tok %.2f  | v: ins %.2f tok %.2f Pk %.2f'%(np.linalg.norm(ins.T@u),np.linalg.norm(Tok.T@u),np.linalg.norm(ins.T@v),np.linalg.norm(Tok.T@v),np.linalg.norm(Pk.T@v)))
```