```python
import sys; sys.path.insert(0,'/work')
from ana import *
P=load_pass(7)
meta=pickle.load(open('/work/meta7.pkl','rb'))
def grid(p):
    lg=P['logits'][p]; x=P['X'][p].reshape(81,3)
    out=[]
    for r in range(9):
        s=''
        for c in range(9):
            i=r*9+c
            if x[i,0]: s+='    #'
            elif x[i,1]: s+='    @'
            else: s+=f'{lg[i]:5.1f}'
        out.append(s)
    return '\n'.join(out)+f'\npass {lg[81]:.1f}'
for p in [1,2,6,9,13,33]:
    print(p, meta[p]); print(grid(p))
    h6=P[6][p]
    if p==2:
        for m in [41,22,57]:
            print(divmod(m,9),'U',[round(float(h6[j]),1) for j in go.NB[m]],'eye',h6[81+m],'SA',h6[162+m],'J',round(float(h6[243+m]),2),'POS',round(float(h6[324+m]),2))
        print('CC active',[(k,round(float(h6[k]),2)) for k in range(405,509) if h6[k]>0])
```