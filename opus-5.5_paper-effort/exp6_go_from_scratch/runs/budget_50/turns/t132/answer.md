```python
s=open('/work/build4.py').read()
old="""            j=n.u(5,nm,p); b[5][j]=c-al*d
            for q in NB[p]:
                W[5][u('O',q),j]+=al;"""
new="""            j=n.u(5,nm,p); b[5][j]=c-al*d+(prm['kap']*(2-d) if nm=='U3' else 0)
            for q in NB[p]:
                if nm=='U3': W[5][u('O',q),j]+=prm['kap']
                W[5][u('O',q),j]+=al;"""
assert old in s
s=s.replace(old,new).replace("komi=7.5,","komi=7.5,kap=0.0,")
open('/work/build4.py','w').write(s)
import sys,json; sys.path.insert(0,'/work'); import build4
BASE=json.load(open('/work/base.json')); BASE.update(scap3=0.75,pw2=0.4,kap=1.0); json.dump(BASE,open('/work/base.json','w'))
prm=build4.default_prm(); prm.update(BASE)
n=build4.build(prm,inp,mv); n.save(save); print(n.used)
```