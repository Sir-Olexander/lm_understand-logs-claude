```python
import numpy as np, pickle, sys
sys.path.insert(0,'/work')
import go
boards=pickle.load(open('/work/testpos2.pkl','rb'))
d='/work/pass_1/'
X=np.load(d+'inputs.npy'); A={k:np.load(d+f'act_L{k}.npy') for k in range(1,7)}
logits=np.load(d+'logits.npy'); probs=np.load(d+'probs.npy')
# check inputs
bad=0
for p,(b,turn) in enumerate(boards):
    own=(b==turn).astype(int); opp=(b==3-turn).astype(int)
    x=X[p].reshape(81,3)
    if not (np.all(x[:,0]==own) and np.all(x[:,1]==opp) and np.all(x[:,2]==(turn==2))): bad+=1
print('input mismatches',bad)
# liberty flags
stats={}
def libs_map(b):
    L=np.zeros(81,int)
    for i in range(81):
        if b[i]: L[i]=len(go.group(b,i)[1])
    return L
from collections import Counter
cA=Counter(); cT=Counter()
for p,(b,turn) in enumerate(boards):
    L=libs_map(b); h5=A[5][p]
    for i in range(81):
        if b[i]==0: 
            for base in (162,243,324,405):
                if h5[base+i]!=0: cA['empty_nonzero']+=1
            continue
        col='own' if b[i]==turn else 'opp'
        a=h5[(162 if col=='own' else 243)+i]; t=h5[(324 if col=='own' else 405)+i]
        lb=min(L[i],3)
        cA[(col,lb,round(float(a),3))]+=1
        cT[(col,lb,round(float(t),3))]+=1
print(sorted(cA.items(),key=str))
print(sorted(cT.items(),key=str))
```