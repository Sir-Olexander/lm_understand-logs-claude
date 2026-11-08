```python
import sys, json, random
sys.path.insert(0,'/work')
import importlib, go; importlib.reload(go)
random.seed(1)
positions=[]; meta=[]
for k in range(200):
    g=go.Game(); L=random.randint(5,120); moves=[]
    for t in range(L):
        lm=[m for m in g.legal_moves() if not go.is_own_eye(g.b,m,g.turn)]
        m=random.choice(lm) if lm else 81
        g.play(m); moves.append(m if m!=81 else "pass")
        if g.passes>=2: break
    # avoid ending with two passes
    if len(moves)>=2 and moves[-1]=="pass" and moves[-2]=="pass": moves=moves[:-1]
    positions.append({"moves":moves})
json.dump(positions,open('/work/testpos1.json','w'))
print(len(positions), sum(len(p['moves']) for p in positions)/200)
```