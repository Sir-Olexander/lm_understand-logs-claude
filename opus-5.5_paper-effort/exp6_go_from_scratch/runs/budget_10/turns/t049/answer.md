```python
code = open('/work/build.py').read()
old="""            u=Fb+i
            W2[Hb+i,u]+=16
            for j in NB[i]: W2[Hb+j,u]+=1
            W2[Sb+i,u]+=20; b2[u]=-36
            u=F3b+i
            W2[Hb+i,u]+=1
            for j in NB[i]: W2[Hb+j,u]+=1
            W2[Sb+i,u]+=20; b2[u]=-22"""
new="""            # exact seeds: libs >= e + ceil(sum_nbr_e / 2)
            u=Fb+i
            W2[Hb+i,u]+=5
            for j in NB[i]: W2[Hb+j,u]+=2
            W2[Sb+i,u]+=40; b2[u]=-45
            u=F3b+i
            W2[Hb+i,u]+=4
            for j in NB[i]: W2[Hb+j,u]+=2
            W2[Sb+i,u]+=40; b2[u]=-49"""
assert old in code
code=code.replace(old,new)
old2="""            W6[j,u]-=1; W6[81+j,u]+=1; W6[324+j,u]+=2; W6[243+j,u]-=3"""
assert old2 in code
code=code.replace(old2,"""            W6[j,u]-=1; W6[81+j,u]+=1; W6[324+j,u]+=2; W6[243+j,u]-=P['sa_kill']""")
code=code.replace("w_eye=30.0, w_sa=15.0,","w_eye=30.0, w_sa=15.0, sa_kill=1.5,")
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
for k,v in W.items(): save(k,v.astype(np.float32))
print('ok', W['L2.W'][162+40,162+40], W['L2.b'][162+40], W['L6.W'][243+41,162+40])
```