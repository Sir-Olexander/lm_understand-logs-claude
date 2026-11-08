```python
w=W['ln1_w.0']; w2=W['ln2_w.0']
V1=W['W_v.0'][64:128]; V5=W['W_v.0'][320:384]
def rs(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
A=rs(V1,30); B=rs(V5,30)
U,cs,Vt=np.linalg.svd(A.T@B)
shared=A@U[:,:22]   # 22 most-shared directions
d=np.sum(shared**2,1)
print("corr shared-diag with ln1_w", np.corrcoef(d,w)[0,1].round(2), "with ln2_w", np.corrcoef(d,w2)[0,1].round(2))
o=np.argsort(-d)
print("top dims by token weight: ln1_w", np.round(w[o[:30]],2))
print("ln2_w", np.round(w2[o[:30]],2))
print("token-diag sorted", np.round(d[o[:60]],3))
print("participation ratio", (d.sum()**2/np.sum(d**2)).round(1))
# check raw W_v of positional-ish heads (H4,H0,H3) row spaces vs ln1_w
for j in (0,3,4):
    Vj=W['W_v.0'][64*j:64*j+64]; R=rs(Vj,4); dj=np.sum(R**2,1)
    print(j,"pos-head V top4 diag corr ln1_w", np.corrcoef(dj,w)[0,1].round(2), "PR", (dj.sum()**2/np.sum(dj**2)).round(1))
```