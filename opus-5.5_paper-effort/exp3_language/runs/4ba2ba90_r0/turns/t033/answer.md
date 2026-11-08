```python
def heads(l):
    g=W[f'ln1_w.{l}']
    res=[]
    for j in range(6):
        s_=slice(64*j,64*j+64)
        Q=W[f'W_q.{l}'][s_]*g; K=W[f'W_k.{l}'][s_]*g; V=W[f'W_v.{l}'][s_]*g; O=W[f'W_o.{l}'][:,s_]
        res.append((Q,K,V,O))
    return res
H={l:heads(l) for l in range(4)}
U,s,_=np.linalg.svd(W['W_proj.3'])
Str=U[:,1:23]; Med=U[:,23:28]
Q,K,V,O=H[3][1]; OV=O@V
def outsplit(vec):
    y=OV@vec; return np.linalg.norm(Str.T@y), np.linalg.norm(Med.T@y), np.linalg.norm(y)
rng=np.random.default_rng(1)
R=[outsplit(x/np.linalg.norm(x)) for x in rng.normal(size=(200,384))]
print('random mean strong %.2f medium %.2f total %.2f'%tuple(np.mean(R,axis=0)))
for j in [0,1,3,4,5]:
    Qa,Ka,Va,Oa=H[0][j]
    uu,ss,vv=np.linalg.svd(Qa.T@Ka)
    for nm,vec in [('u',uu[:,0]),('v',vv[0])]:
        print(f'L0H{j} {nm}: strong %.2f medium %.2f total %.2f'%outsplit(vec))
# best input directions of OV for medium output
um,sm,vtm=np.linalg.svd(Med.T@OV)
print('max gain into medium', np.round(sm,2))
ust,sst,vst=np.linalg.svd(Str.T@OV); print('max gain into strong', np.round(sst[:5],2))
```