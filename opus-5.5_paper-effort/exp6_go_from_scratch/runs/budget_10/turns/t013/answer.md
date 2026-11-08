```python
import sys; sys.path.insert(0,'/work')
from ana import *
boards=pickle.load(open('/work/testpos2.pkl','rb'))
P=load_pass(1)
for p in [8,17,2,9]:
    b,turn=boards[p]
    print(p,'to move','X' if turn==1 else 'O')
    print(go.show(b))
    lg=P['logits'][p]; leg=P['legal'][p]
    order=np.argsort(-np.where(leg,lg,-1e9))[:8]
    print([(divmod(int(m),9) if m<81 else 'pass', round(float(lg[m]),2)) for m in order])
    h6=P[6][p]
    for m in order[:4]:
        if m<81: print(divmod(int(m),9),'U-nb',[round(float(h6[j]),1) for j in go.NB[m]],'eye',h6[81+m],'SA',h6[162+m],'J',round(float(h6[243+m]),2),'OT',round(float(h6[324+m]),2),'XT',round(float(h6[405+m]),2))
    print('NST',h6[486])
```