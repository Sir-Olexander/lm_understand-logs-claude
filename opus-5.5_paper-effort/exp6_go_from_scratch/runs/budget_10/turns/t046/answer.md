```python
import json, numpy as np, sys
sys.path.insert(0,'/work')
import go
G=json.load(open('/work/pass_5/games.json'))
B=np.load('/work/pass_5/boards.npy')
def conv(bd):
    b=np.zeros(81,np.int8); b[bd==1]=1; b[bd==-1]=2; return b
def show(b,mark=None):
    return '\n'.join(''.join('*' if r*9+c==mark else '.XO'[b[r*9+c]] for c in range(9)) for r in range(9))
R2={i:[j for j in range(81) if 0<abs(i//9-j//9)+abs(i%9-j%9)<=2] for i in range(81)}
cases=[(3,74),(7,64),(8,38),(10,74),(25,60),(28,74),(45,52),(56,45),(58,87),(60,61),(75,67),(85,63),(95,43),(91,81)]
for gi,k in cases:
    g=G[gi]; me=1 if g['net_color']=='black' else 2
    b=conv(B[gi,k-1]); p=g['moves'][k]['move']
    e=sum(1 for j in go.NB[p] if b[j]==0)
    info=[]
    for j in go.NB[p]:
        if b[j]:
            G_,L=go.group(b,j); dens=sum(1 for t in R2[j] if b[t]==b[j])
            info.append(('own' if b[j]==me else 'opp',len(G_),len(L),dens))
    print(gi,k,divmod(p,9),'e',e,info,[(divmod(a,9) if a<81 else 'P',round(pp,2)) for a,pp in g['moves'][k]['top5'][:3]])
    print(show(b,p))
```