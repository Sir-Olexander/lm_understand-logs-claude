```python
s=open('/work/build4.py').read()
old="""    for k in (2,3,4):
        for nm in ['S','O','NS','NO','US','UO']: n.alloc(k,nm)"""
new="""    for k in (2,3,4):
        for nm in (['S','O','NS','NO','US','UO'] if (k<4 or not prm['iso5']) else ['S','O','NS','NO','US','Iso']): n.alloc(k,nm)"""
assert old in s; s=s.replace(old,new)
old="""            for X,col,Ev in (('N','S','Ev'),('N','O','Ev'),('U','S','E3'),('U','O','E3')):"""
new="""            if k==4 and prm['iso5']:
                j=n.u(4,'Iso',p); d=len(NB[p]); b[4][j]=4-2*d
                for q in NB[p]: W[4][n.u(3,'NO',q),j]-=1; W[4][n.u(3,'S',q),j]-=3; W[4][n.u(3,'O',q),j]+=2
                W[4][n.u(3,'S',p),j]-=10; W[4][n.u(3,'O',p),j]-=10
            for X,col,Ev in ((('N','S','Ev'),('N','O','Ev'),('U','S','E3'),('U','O','E3')) if (k<4 or not prm['iso5']) else (('N','S','Ev'),('N','O','Ev'),('U','S','E3'))):"""
assert old in s; s=s.replace(old,new)
old="""            W[5][u('UO',q),j]+=prm['atk']; W[5][u('NO',q),j]+=prm['cap']-prm['atk']"""
new="""            if prm['iso5']: W[5][u('NO',q),j]+=prm['cap']
            else: W[5][u('UO',q),j]+=prm['atk']; W[5][u('NO',q),j]+=prm['cap']-prm['atk']
        if prm['iso5']: W[5][u('Iso',p),j]-=prm['iso']"""
assert old in s; s=s.replace(old,new)
s=s.replace("komi=7.5,","komi=7.5,iso5=False,")
open('/work/build4.py','w').write(s)
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json'))
prm=build4.default_prm(); prm.update(BASE); prm.update(iso5=True,iso=6)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```