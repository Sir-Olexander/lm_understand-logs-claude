```python
s=open('/work/build3.py').read()
s=s.replace("for nm in ['S','O','EvS','EvO','E3S']: n.alloc(1,nm)","for nm in ['S','O','EvS','EvO','E3S','E3O']: n.alloc(1,nm)")
s=s.replace("""        j=n.u(1,'E3S',p)
        for m in [p]+NB[p]: W[1][n.u(0,'LS',m),j]+=1
        W[1][n.u(0,'S',p),j]+=20; b[1][j]=-22""","""        for col in 'SO':
            j=n.u(1,'E3'+col,p)
            for m in [p]+NB[p]: W[1][n.u(0,'L'+col,m),j]+=1
            W[1][n.u(0,col,p),j]+=20; b[1][j]=-22""")
s=s.replace("for nm in ['S','O','NS','NO','US']: n.alloc(k,nm)","for nm in ['S','O','NS','NO','US','UO']: n.alloc(k,nm)")
s=s.replace("(('N','S','Ev'),('N','O','Ev'),('U','S','E3'))","(('N','S','Ev'),('N','O','Ev'),('U','S','E3'),('U','O','E3'))")
# Terr moves to L6 (from L5 S,O,NS)
s=s.replace("""    n.alloc(4,'Terr')
    for p in range(81):
        j=n.u(4,'Terr',p); b[4][j]=-prm['terr_th']
        for q in NB[p]: W[4][n.u(3,'S',q),j]+=1; W[4][n.u(3,'O',q),j]-=2; W[4][n.u(3,'NS',q),j]-=10
        for q in range(81):
            if D(p,q)==2: W[4][n.u(3,'S',q),j]+=.5; W[4][n.u(3,'O',q),j]-=1
""","")
s=s.replace("for nm in ['Eye','Pos','U2','U3','NS6','NO6']: n.alloc(5,nm)","for nm in ['Eye','Pos','U2','U3','NS6','Terr']: n.alloc(5,nm)")
s=s.replace("""        for col,w in (('S',prm['esc']),('O',prm['cap'])):""","""        j=n.u(5,'Terr',p); b[5][j]=-prm['terr_th']
        for q in NB[p]: W[5][u('S',q),j]+=1; W[5][u('O',q),j]-=2; W[5][u('NS',q),j]-=10
        for q in range(81):
            if D(p,q)==2: W[5][u('S',q),j]+=.5; W[5][u('O',q),j]-=1
        W[6][j,o]=-prm['terr']
        for col,w in (('S',prm['esc']),):""")
s=s.replace("""        W[5][u('Terr',p),j]-=prm['terr']
""","")
s=s.replace("""            W[5][u('US',q),j]+=prm['def']; W[5][u('NS',q),j]-=prm['def']""","""            W[5][u('US',q),j]+=prm['def']; W[5][u('NS',q),j]-=prm['def']
            W[5][u('UO',q),j]+=prm['atk']; W[5][u('NO',q),j]+=prm['cap']-prm['atk']""")
# U2m = relu(1-0.9A-0.3T), M = relu(2.5-A-0.5T);  A=deg-sO+sS-2sUS+scap*sNO, T=sUS-sNS
old=s[s.index("        for nm,th,wU,wN in (('U2',1,2,0),('U3',2,0,2)):"):s.index("    b[6][81]=prm['pass']")]
new="""        for nm,c,al,be in (('U2',1,.9,.3),('U3',2.5,1,.5)):
            j=n.u(5,nm,p); b[5][j]=c-al*d
            for q in NB[p]:
                W[5][u('O',q),j]+=al; W[5][u('S',q),j]-=al; W[5][u('US',q),j]+=2*al-be; W[5][u('NS',q),j]+=be; W[5][u('NO',q),j]-=al*prm['scap']
            W[6][j,o]=-prm[nm]
"""
s=s.replace(old,new)
s=s.replace("U2=25,U3=25,","U2=250,U3=6,atk=4,").replace("**{'def':2}","**{'def':4}")
open('/work/build4.py','w').write(s)
import sys; sys.path.insert(0,'/work'); import build4
n=build4.build(build4.default_prm(),inp,mv); n.save(save); print(n.used)
```