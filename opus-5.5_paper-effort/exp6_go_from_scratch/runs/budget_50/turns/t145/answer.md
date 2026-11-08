```python
s=open('/work/build4.py').read()
old="""        for col,w in (('S',prm['esc']),):"""
new="""        if prm['iso6']:
            j=n.u(5,'NS6',p); b[5][j]=4-2*d
            for q in NB[p]: W[5][u('NO',q),j]-=1; W[5][u('S',q),j]-=3; W[5][u('O',q),j]+=2
            W[5][u('S',p),j]-=10; W[5][u('O',p),j]-=10
            W[6][j,o]=-prm['iso']
        for col,w in ((('S',prm['esc']),) if not prm['iso6'] else ()):"""
assert old in s; s=s.replace(old,new)
old="""            W[5][u('US',q),j]+=prm['def']; W[5][u('NS',q),j]-=prm['def']"""
new="""            W[5][u('US',q),j]+=prm['def']; W[5][u('NS',q),j]-=prm['def']
            if prm['iso6']: W[5][u('NS',q),j]+=prm['esc']"""
assert old in s; s=s.replace(old,new)
s=s.replace("komi=7.5,","komi=7.5,iso6=False,")
open('/work/build4.py','w').write(s)
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json')); BASE['atk']=6; json.dump(BASE,open('/work/base.json','w'))
prm=build4.default_prm(); prm.update(BASE); prm.update(iso6=True,iso=6,eye=20)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```