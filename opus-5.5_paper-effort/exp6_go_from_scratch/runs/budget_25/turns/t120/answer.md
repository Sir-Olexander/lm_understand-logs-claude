```python
import importlib, sys; sys.path.insert(0,'/work')
import v16_ab; importlib.reload(v16_ab)
import numpy as np, pickle
P,_=pickle.load(open('/work/v16_units.pkl','rb'))
net,U=v16_ab.build(P)
names=[(k,v) for k,v in net.names[0].items() if isinstance(k,tuple) and k[0]=='cert']
print(len(names), names[:5])
k,jj=names[0]
W=net.W[0]; nz=np.nonzero(W[:,jj])[0]
print([(i//3//9,i//3%9,i%3,W[i,jj]) for i in nz], net.B[0][jj])
# check L2 usage: find unit x at L2 for middle stone
s=k[1]
for kk in (0,1):
    j2=U[('x',2,s,kk)]
    print('L2 x',s,kk,'weight from cert',net.W[1][jj,j2])
print(saved())
```