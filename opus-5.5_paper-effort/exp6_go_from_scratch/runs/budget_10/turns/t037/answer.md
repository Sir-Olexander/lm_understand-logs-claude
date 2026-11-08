```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_4/games.json'))
B=np.load('/work/pass_4/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
tot=0; blunder=0; sizes=[]; ex=[]
for gi,g in enumerate(G):
    netcol = -1 if g['net_color']=='white' else 1
    mv=g['moves']
    for k in range(len(mv)-1):
        m=mv[k]
        if m['by']!='net' or m['move']==81: continue
        tot+=1
        p=m['move']
        if B[gi,k+1][p]!=netcol:  # captured on next ply
            lost=int(((B[gi,k]==netcol)&(B[gi,k+1]!=netcol)).sum())
            blunder+=1; sizes.append(lost); ex.append((gi,k,lost, m['top5'][0][1]))
print(tot,blunder,sorted(sizes))
print(ex[:30])
```