```python
code = r'''
import numpy as np
NB=[[] for _ in range(81)]
DG=[[] for _ in range(81)]
for r in range(9):
    for c in range(9):
        s=r*9+c
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: NB[s].append(rr*9+cc)
        for dr,dc in ((-1,-1),(-1,1),(1,-1),(1,1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<9 and 0<=cc<9: DG[s].append(rr*9+cc)
EDGES=[(s,t) for s in range(81) for t in NB[s] if s<t]
def IO(s): return (s//9*9+s%9)*3+0
def IP(s): return (s//9*9+s%9)*3+1
IW=2
class Net:
    def __init__(self):
        self.W=[np.zeros((243,512),np.float64)]+[np.zeros((512,512),np.float64) for _ in range(5)]+[np.zeros((512,82),np.float64)]
        self.B=[np.zeros(512,np.float64) for _ in range(6)]+[np.zeros(82,np.float64)]
        self.n=[0]*6
        self.names=[dict() for _ in range(6)]
    def unit(self,layer,name,terms,bias=0.0):
        j=self.n[layer-1]
        assert j<512, ('layer full',layer)
        self.n[layer-1]+=1
        for src,w in terms:
            self.W[layer-1][src,j]+=w
        self.B[layer-1][j]+=bias
        self.names[layer-1][name]=j
        return j
    def outw(self,k,terms,bias=0.0):
        for src,w in terms: self.W[6][src,k]+=w
        self.B[6][k]+=bias
    def save_all(self,save):
        nm=['L1','L2','L3','L4','L5','L6','out']
        for i in range(7):
            save(nm[i]+'.W',self.W[i].astype(np.float32))
            save(nm[i]+'.b',self.B[i].astype(np.float32))
'''
open('/work/netlib_ab.py','w').write(code)
print('ok')
```