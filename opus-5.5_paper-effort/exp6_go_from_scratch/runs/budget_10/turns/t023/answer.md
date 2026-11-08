```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_2/games.json'))
B=np.load('/work/pass_2/boards.npy')
def show(bd):
    ch={0:'.',1:'X',-1:'O'}
    return '\n'.join(''.join(ch[int(bd[r*9+c])] for c in range(9)) for r in range(9))
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
for gi,ks in [(29,[93,95,97,99,101,103]),(36,[83,85])]:
    for k in ks:
        print(gi,k); b=conv(B[gi,k]); print(show(B[gi,k])); print(go.score(b))
```