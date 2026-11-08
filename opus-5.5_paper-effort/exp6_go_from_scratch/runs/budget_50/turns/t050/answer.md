```python
code=r'''
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
'''
open('/work/ana.py','a').write(code)
import sys; sys.path.insert(0,'/work'); from ana import *
print(diag(11))
```