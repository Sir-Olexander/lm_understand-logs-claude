```python
import sys; sys.path.insert(0,'/work')
from ana import *
P=load_pass(3)
meta=pickle.load(open('/work/meta3.pkl','rb'))
def grid(p):
    lg=P['logits'][p]; leg=P['legal'][p]; x=P['X'][p].reshape(81,3)
    out=[]
    for r in range(9):
        s=''
        for c in range(9):
            i=r*9+c
            if x[i,0]: s+='   #'
            elif x[i,1]: s+='   @'
            else: s+=f'{lg[i]:4.1f}'
        out.append(s)
    return '\n'.join(out)+f'\npass {lg[81]:.1f}'
for p in [0,1,2,5,7,8,17,21]:
    print(p, meta[p]); print(grid(p))
```