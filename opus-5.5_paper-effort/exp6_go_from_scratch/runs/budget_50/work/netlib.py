
import numpy as np
P=[(r,c) for r in range(9) for c in range(9)]
def idx(r,c): return r*9+c
def nb(p):
    r,c=divmod(p,9); out=[]
    for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
        rr,cc=r+dr,c+dc
        if 0<=rr<9 and 0<=cc<9: out.append(rr*9+cc)
    return out
NB=[nb(p) for p in range(81)]
def ring2(p):
    r,c=divmod(p,9)
    return [(rr*9+cc) for rr in range(9) for cc in range(9) if abs(rr-r)+abs(cc-c)==2]
R2=[ring2(p) for p in range(81)]
class Net:
    def __init__(s):
        s.W=[np.zeros((243,512),np.float32)]+[np.zeros((512,512),np.float32) for _ in range(5)]+[np.zeros((512,82),np.float32)]
        s.b=[np.zeros(512,np.float32) for _ in range(6)]+[np.zeros(82,np.float32)]
        s.used=[0]*6; s.blk=[{} for _ in range(6)]
    def alloc(s,k,name,n=81):
        i=s.used[k]; s.used[k]+=n; assert s.used[k]<=512,(k,name)
        s.blk[k][name]=i; return i
    def u(s,k,name,p): return s.blk[k][name]+p
    def save(s,save_fn):
        names=['L%d'%i for i in range(1,7)]+['out']
        for i,nm in enumerate(names):
            save_fn(nm+'.W',s.W[i]); save_fn(nm+'.b',s.b[i])
