```python
import sys; sys.path.insert(0,'/work')
from ana import *
boards=pickle.load(open('/work/testpos2.pkl','rb'))
P=load_pass(1)
shown=0
for p,(b,turn) in enumerate(boards):
    L=libs_map(b); h5=P[5][p]
    done=set()
    for col,base in ((turn,162),(3-turn,243)):
        S=seeds(b,col)
        for i in range(81):
            if b[i]==col and L[i]>=2 and h5[base+i]>0 and i not in done:
                g,l=go.group(b,i)
                done|=g
                if not (g & S) and shown<8:
                    shown+=1
                    bb=b.copy()
                    print('pos',p,'group size',len(g),'libs',sorted(l))
                    rows=[]
                    for r in range(9):
                        s=''
                        for c in range(9):
                            k=r*9+c
                            s+= ('#' if k in g else '*' if k in l else '.XO'[b[k]])
                        rows.append(s)
                    print('\n'.join(rows))
```