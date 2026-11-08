```python
code=r'''
def nbrs(p):
    r,c=divmod(p,9); o=[]
    if r>0:o.append(p-9)
    if r<8:o.append(p+9)
    if c>0:o.append(p-1)
    if c<8:o.append(p+1)
    return o
NBL=[nbrs(p) for p in range(81)]
def groups(bd):
    seen=set(); out=[]
    for p in range(81):
        if bd[p]==0 or p in seen: continue
        col=bd[p]; st=[p]; seen.add(p); libs=set(); k=0
        while k<len(st):
            q=st[k]; k+=1
            for m in NBL[q]:
                if bd[m]==0: libs.add(m)
                elif bd[m]==col and m not in seen: seen.add(m); st.append(m)
        out.append((col,st,libs))
    return out
def play(bd,p,col):
    bd=bd.copy(); bd[p]=col; cap=0
    for col2,st,libs in groups(bd):
        if col2==-col and not libs:
            for q in st: bd[q]=0
            cap+=len(st)
    return bd,cap
def board_before(B,i,ply):
    return np.zeros(81,np.int8) if ply==0 else B[i,ply-1]
'''
open('/work/ana.py','a').write(code)
import sys; sys.path.insert(0,'/work'); from ana import *
G,B=load(2)
stats=dict(cap_avail=0,cap_taken=0,cap_size_missed=0,inatari=0,saved=0,selfatari=0,selfatari_size=0,moves=0)
for i,g in enumerate(G):
    if g['opponent']!='strong': continue
    me=1 if g['net_color']=='black' else -1
    for m in g['moves']:
        if m['by']!='net' or m['move']==81: continue
        bd=board_before(B,i,m['ply']); gr=groups(bd); stats['moves']+=1
        caps=[(st,l) for c,st,l in gr if c==-me and len(l)==1]
        if caps:
            stats['cap_avail']+=1; libs={list(l)[0] for st,l in caps}
            if m['move'] in libs: stats['cap_taken']+=1
            else: stats['cap_size_missed']+=max(len(st) for st,l in caps)
        mine=[(st,l) for c,st,l in gr if c==me and len(l)==1]
        if mine: stats['inatari']+=1
        nb,cap=play(bd,m['move'],me)
        for c,st,l in groups(nb):
            if m['move'] in st and len(l)==1 and cap==0: stats['selfatari']+=1; stats['selfatari_size']+=len(st)
print(stats)
```