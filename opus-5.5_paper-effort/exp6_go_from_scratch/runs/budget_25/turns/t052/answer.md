```python
src=open('/work/v4_ab.py').read()
# 1. Merge eye into U0
old="""        U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*1.5-1.0)
        U[('eye',6,p)]=net.unit(6,('eye',p),[(U[('O',5,n)],1.0) for n in NB[p]],-(deg-1))"""
new="""        t0+=[(U[('O',5,n)],P['EYEU']) for n in NB[p]]
        U[('U0',6,p)]=net.unit(6,('U0',p),t0,P['B0']*1.5-1.0-P['EYEU']*(deg-1))"""
assert old in src; src=src.replace(old,new)
# 2. positional: replace pos aggregator with P1/P2
i0=src.index("        # positional aggregator"); i1=src.index("    # ---- output")
newpos='''        # positional: A_o, A_x
        r,c=divmod(p,9); Ao=[]; Ax=[]; axb=0.0
        for t in range(81):
            dd=abs(t//9-r)+abs(t%9-c)
            if dd==0 or dd>3 or t not in BSUB and False: pass
            if dd==0 or dd>3: continue
            w=P['kern'][dd]
            Ao.append((U[('O',5,t)],w))
            if t in BSUB:
                Ax+=[(U[('O',5,t)],w),(U[('b',5,t)],-w)]; axb+=w
        lam,mu=P['lam'],P['mu']
        t1=[(i,(lam-mu)*w) for i,w in Ax]+[(i,-mu*w) for i,w in Ao]
        U[('P1',6,p)]=net.unit(6,('P1',p),t1,P['C']+(lam-mu)*axb)
        t2=[(i,w) for i,w in Ax]+[(i,-w) for i,w in Ao]
        U[('P2',6,p)]=net.unit(6,('P2',p),t2,axb)
    U[('gS',6)]=net.unit(6,'gS',[(U[('gS',5)],1.0)],0.0)
    U[('gD',6)]=net.unit(6,'gD',[(U[('gD',5)],1.0)],0.0)
'''
src=src[:i0]+newpos+src[i1:]
# output
old="""        terms+=[(U[('U1',6,p)],-P['SA1']),(U[('U0',6,p)],-P['SA0']),(U[('eye',6,p)],-P['EYE']),(U[('pos',6,p)],1.0)]
        net.outw(p,terms,-P['C']+P['LV'][line(p)])
    net.outw(81,[],P['PASS'])"""
new="""        terms+=[(U[('U1',6,p)],-P['SA1']),(U[('U0',6,p)],-P['SA0']),(U[('P1',6,p)],1.0),(U[('P2',6,p)],-P['lam'])]
        net.outw(p,terms,-P['C']+P['LV'][line(p)])
    net.outw(81,[(U[('gS',6)],P['kS']),(U[('gD',6)],P['kD'])],P['PASS'])"""
assert old in src; src=src.replace(old,new)
# globals at L1..L5
old="""    # ---- L2..L4"""
new="""    U[('gS',1)]=net.unit(1,'gS',[(IO(s),1.0) for s in range(81)]+[(IP(s),1.0) for s in range(81)],-P['S0'])
    U[('gD',1)]=net.unit(1,'gD',[(IO(s),1.0) for s in range(81)]+[(IP(s),-1.0) for s in range(81)],-P['D0'])
    for l in (2,3,4,5):
        for g in ('gS','gD'):
            U[(g,l)]=net.unit(l,g,[(U[(g,l-1)],1.0)],0.0)
    # ---- L2..L4"""
assert old in src; src=src.replace(old,new)
src=src.replace("SKIPP=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44])","SKIPP=set([0,8,72,80, 2,4,6, 18,36,54, 74,76,78, 26,44, 62,10])")
open('/work/v5_ab.py','w').write(src)
import importlib, sys; sys.path.insert(0,'/work')
import v5_ab; importlib.reload(v5_ab)
P=dict(eps=1e-5,CAP=20.0,ESC=16.0,B1=4.0,BIG1=10.0,B0=24.0,BIG0=12.0,BIGC=100.0,SA1=5.0,SA0=1.5,EYEU=80.0,C=30.0,
       LV={1:-4.0,2:-1.0,3:1.0,4:1.5,5:2.0},kern={1:1.0,2:0.6,3:0.3},lam=3.0,mu=0.5,
       S0=50.0,D0=10.0,kS=0.5,kD=0.5,PASS=-12.0)
net,U=v5_ab.build(P)
print(net.n)
net.save_all(save)
import pickle; pickle.dump((P,{str(k):v for k,v in U.items()}),open('/work/v5_units.pkl','wb'))
```