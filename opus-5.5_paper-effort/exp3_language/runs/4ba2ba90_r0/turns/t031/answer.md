```python
U,s,Vt=np.linalg.svd(W['W_proj.3'])
print([ (i,round(float(s[i]),2), round(float(1/np.sum(U[:,i]**4)),0)) for i in range(26,50)])
# same for W_o.3 and for combined late writers
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s_=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s_]*g; K=W[f'W_k.{l}'][s_]*g; V=W[f'W_v.{l}'][s_]*g; O=W[f'W_o.{l}'][:,s_]
        res.append((Q,K,V,O))
    return res
H3=heads(3)
Q,K,V,O=H3[1]; OV=O@V
u,sv,vt=np.linalg.svd(OV)
print('L3H1 OV', [(i,round(float(sv[i]),2), round(float(1/np.sum(u[:,i]**4)),0)) for i in range(0,34)])
# overlap of L3H1 OV output dirs with W_proj3 dense dirs 1..27 and medium 23..27
D=U[:,1:28]; Med=U[:,23:28]; Str=U[:,1:23]
for i in range(32):
    print(i, round(float(sv[i]),2), 'in strong %.2f in medium %.2f'%(np.linalg.norm(Str.T@u[:,i])**2, np.linalg.norm(Med.T@u[:,i])**2))
```