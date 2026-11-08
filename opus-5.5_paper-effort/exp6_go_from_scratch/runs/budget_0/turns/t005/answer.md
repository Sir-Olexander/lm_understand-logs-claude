```python
import pickle, numpy as np
spec=pickle.load(open('/work/spec.pkl','rb'))
layers=spec['layers']
names=[]
inv0={}
for r in range(9):
    for c in range(9):
        inv0[inp(r,c,0)]=('xo',(r,c)); inv0[inp(r,c,1)]=('xp',(r,c)); inv0[inp(r,c,2)]=('white',(r,c))
invs=[inv0]+[{j:k for j,(k,_,_) in enumerate(units)} for _,units in layers]
def show(li, key):
    nm=['L1','L2','L3','L4','L5','L6'][li]
    Wm=load(nm+'.W'); b=load(nm+'.b')
    j=[j for j,(k,_,_) in enumerate(layers[li][1]) if k==key][0]
    nz=np.nonzero(Wm[:,j])[0]
    print(nm,key,'bias',b[j],sorted([(invs[li][i],float(Wm[i,j])) for i in nz]))
show(0,('Zown',(0,0)))
show(0,('Eopp',(4,4)))
show(1,('Sown',(0,0)))
show(1,('VSopp',(0,4)))
show(2,('U1opp',(0,0)))
show(3,('NV2own',(8,8)))
show(4,('SOFT',(0,1)))
show(5,('S',(4,4)))
show(5,('IN',(0,0)))
Wo=load('out.W'); bo=load('out.b')
print(bo[:3],bo[81], np.unique(Wo))
print({k:int((Wo[[j for j,(kk,_,_) in enumerate(layers[5][1]) if kk[0]==k],:]!=0).sum()) for k in ['S','ESCX','HARD','EYE','IN','ALLOPP']})
```