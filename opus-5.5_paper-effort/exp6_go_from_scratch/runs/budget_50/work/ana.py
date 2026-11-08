
import json,numpy as np
def load(k):
    G=json.load(open('/work/pass_%d/games.json'%k)); B=np.load('/work/pass_%d/boards.npy'%k)
    return G,B
def show(bd):
    return '\n'.join(''.join('.XO'[v] for v in bd[r*9:(r+1)*9]) for r in range(9))
def showmany(bds):
    rows=[show(b).split('\n') for b in bds]
    return '\n'.join('  '.join(r[i] for r in rows) for i in range(9))

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

def diag(k,opp='strong'):
    from collections import Counter
    G,B=load(k); st=Counter()
    for i,g in enumerate(G):
        if g['opponent']!=opp: continue
        me=1 if g['net_color']=='black' else -1
        st['games']+=1; st['wins']+=g['result']>0
        if g['end']=='repetition': st['rep_by_'+g['moves'][-1]['by']]+=1; st['rep_lost_ahead']+= (g['margin']>0 and g['result']<0)
        for m in g['moves']:
            if m['by']!='net': continue
            st['moves']+=1
            if m['move']==81: st['pass']+=1; continue
            p=m['move']; bd=board_before(B,i,m['ply']); gr=groups(bd)
            sav=set(); capl=set()
            for c,stn,l in gr:
                if len(l)!=1: continue
                L=list(l)[0]
                if c==-me: capl.add(L)
                elif len(stn)>=2:
                    nb,cap=play(bd,L,me)
                    if len([ll for cc,s2,ll in groups(nb) if L in s2][0])>=2: sav.add(L)
            if capl: st['cap_avail']+=1; st['cap_taken']+=p in capl
            if sav: st['savable']+=1; st['saved']+=p in sav; st['saved_or_cap']+= (p in sav or p in capl)
            nb,cap=play(bd,p,me)
            s2,ll=[(s2,ll) for cc,s2,ll in groups(nb) if p in s2][0]
            if cap==0 and len(ll)==1: st['selfatari']+=1; st['selfatari_sz']+=len(s2)
    return dict(st)
