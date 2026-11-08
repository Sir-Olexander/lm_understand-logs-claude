```python
code=open('/work/build.py').read()
# seed modes
old="""            # exact seeds: libs >= e + ceil(sum_nbr_e / 2)
            u=Fb+i
            W2[Hb+i,u]+=5
            for j in NB[i]: W2[Hb+j,u]+=2
            W2[Sb+i,u]+=40; b2[u]=-45
            u=F3b+i
            W2[Hb+i,u]+=4
            for j in NB[i]: W2[Hb+j,u]+=2
            W2[Sb+i,u]+=40; b2[u]=-49"""
new="""            if P.get('seed','mix')=='old':
                u=Fb+i
                W2[Hb+i,u]+=16
                for j in NB[i]: W2[Hb+j,u]+=1
                W2[Sb+i,u]+=20; b2[u]=-36
                u=F3b+i
                W2[Hb+i,u]+=1
                for j in NB[i]: W2[Hb+j,u]+=1
                W2[Sb+i,u]+=20; b2[u]=-22
            else:
                # exact F seeds: libs >= e + ceil(sum_nbr_e / 2)
                u=Fb+i
                W2[Hb+i,u]+=5
                for j in NB[i]: W2[Hb+j,u]+=2
                W2[Sb+i,u]+=40; b2[u]=-45
                u=F3b+i
                if P.get('seed','mix')=='exact':
                    W2[Hb+i,u]+=4
                    for j in NB[i]: W2[Hb+j,u]+=2
                    W2[Sb+i,u]+=40; b2[u]=-49
                else:   # mix: 1.5e + sum >= 4
                    W2[Hb+i,u]+=3
                    for j in NB[i]: W2[Hb+j,u]+=2
                    W2[Sb+i,u]+=40; b2[u]=-47"""
assert old in code; code=code.replace(old,new)
code=code.replace("tengen=1.5,","tengen=0.4, seed='mix',")
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
print(build.DEF['seed'],build.DEF['tengen'], W['L2.W'][162+324+40-162, 324+40] if False else W['L2.W'][162+40,324+40], W['L2.b'][324+40])
```