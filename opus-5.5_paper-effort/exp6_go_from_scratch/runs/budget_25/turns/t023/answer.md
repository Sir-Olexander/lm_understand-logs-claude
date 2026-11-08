```python
import sys; sys.path.insert(0,'/work')
import gorules_ab as G
import json, numpy as np, pickle
P,Us=pickle.load(open('/work/v2_units.pkl','rb'))
items=json.load(open('/work/pos2_items.json'))
A6=np.load('/work/pass_2/act_L6.npy'); LG=np.load('/work/pass_2/logits.npy'); PR=np.load('/work/pass_2/probs.npy')
def U(key): return Us[str(key)]
for i in [5,6,8,11,12,14,15,23,26,48,56]:
    it=items[i]
    print(i,it['to_move']); print('\n'.join(it['board']))
    top=np.argsort(-PR[i])[:4]
    for p in top:
        if p==81: print('  pass',PR[i,p]); continue
        cap=(A6[i,U(('b',6,p))]-1)/P['eps']
        esc=sum(A6[i,U(('at',6,q))] for q in G.NB[p])*P['ESC']
        print('  (%d,%d) pr=%.3f logit=%.2f cap=%.1f esc=%.1f U1=%.1f U0=%.1f eye=%.0f'%(p//9,p%9,PR[i,p],LG[i,p],cap,esc,A6[i,U(('U1',6,p))],A6[i,U(('U0',6,p))],A6[i,U(('eye',6,p))]))
```