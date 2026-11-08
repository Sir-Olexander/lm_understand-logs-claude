```python
import json, numpy as np
d=json.load(open('/work/pass_1/games.json'))
B=np.load('/work/pass_1/boards.npy')
def show(b):
    ch={0:'.',1:'X',-1:'O'}
    return [''.join(ch[int(b[r*9+c])] for c in range(9)) for r in range(9)]
for gi in [34,66]:
    print(gi,d[gi]['net_color'],d[gi]['margin'])
    rows=[show(B[gi,k]) for k in (9,19,29,39,49,69)]
    for r in range(9): print('   '.join(x[r] for x in rows))
    print([(m['by'][0],m['move']) for m in d[gi]['moves'][:30]])
```