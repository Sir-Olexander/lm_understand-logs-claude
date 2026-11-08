```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
def show(b,mark=None):
    out=[]
    for r in range(9):
        s=''
        for c in range(9):
            i=r*9+c
            s+= '*' if i==mark else '.XO'[b[i]]
        out.append(s)
    return '\n'.join(out)
cases=[(5,58),(16,50),(18,68),(34,58),(49,65),(52,69),(61,51),(27,64),(46,42),(64,87)]
for gi,k in cases:
    g=G[gi]; me=1 if g['net_color']=='black' else 2; op=3-me
    b=conv(B[gi,k-1]) if k>0 else np.zeros(81,np.int8)
    p=g['moves'][k]['move']
    e=sum(1 for j in go.NB[p] if b[j]==0)
    info=[]
    for j in go.NB[p]:
        if b[j]:
            G_,L=go.group(b,j); info.append(('own' if b[j]==me else 'opp',len(G_),len(L)))
    print(gi,k,divmod(p,9),'e',e,info, g['moves'][k]['top5'][:3])
    print(show(b,p))
```