```python
import numpy as np, pickle
spec=pickle.load(open('/work/spec.pkl','rb'))
layers=spec['layers']; C=spec['C']; OUTW=spec['OUTW']
def in_index_L1(key):
    t,q=key; r,c=q
    return inp(r,c,0) if t=='xo' else inp(r,c,1)
prev_idx=None
mats={}
for li,(name,units) in enumerate(layers):
    nin = 243 if li==0 else 512
    Wm=np.zeros((nin,512),np.float32); b=np.zeros(512,np.float32)
    for j,(key,bias,w) in enumerate(units):
        b[j]=bias
        for k,v in w.items():
            i = in_index_L1(k) if li==0 else prev_idx[k]
            Wm[i,j]+=v
    mats[name]=(Wm,b)
    prev_idx={key:j for j,(key,_,_) in enumerate(units)}
# output
Wo=np.zeros((512,82),np.float32); bo=np.zeros(82,np.float32)
for key,j in prev_idx.items():
    t,q=key; p=q[0]*9+q[1]
    Wo[j,p]+=OUTW[t]
for r in range(9):
    for c in range(9): bo[mv(r,c)]=-C
bo[PASS]=spec['PASS']
mats['out']=(Wo,bo)
for nm,(Wm,b) in mats.items():
    assert Wm.shape==tuple(SHAPES[nm+'.W']) and b.shape==tuple(SHAPES[nm+'.b'])
    save(nm+'.W',Wm); save(nm+'.b',b)
print(saved())
# interval analysis on weights: value ranges of each unit (inputs in [0,1])
lo=np.zeros(243); hi=np.ones(243)
for nm in ['L1','L2','L3','L4','L5','L6']:
    Wm,b=load(nm+'.W'),load(nm+'.b')
    plo=b+np.minimum(Wm,0).T@hi+np.maximum(Wm,0).T@lo
    phi=b+np.maximum(Wm,0).T@hi+np.minimum(Wm,0).T@lo
    lo,hi=np.maximum(plo,0),np.maximum(phi,0)
    print(nm,'max hi',hi.max())
    if nm=='L6':
        units=layers[5][1]
        Sidx=[j for j,(k,_,_) in enumerate(units) if k[0]=='S']
        print('S pre-activation min over interval:',plo[Sidx].min())
```