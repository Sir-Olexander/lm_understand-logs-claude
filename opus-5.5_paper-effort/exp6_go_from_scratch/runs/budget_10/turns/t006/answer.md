```python
import sys, json, random
sys.path.insert(0,'/work')
import go
random.seed(2)
positions=[]; boards=[]
for k in range(128):
    g=go.Game(); L=random.randint(10,130)
    for t in range(L):
        lm=[m for m in g.legal_moves() if not go.is_own_eye(g.b,m,g.turn)]
        m=random.choice(lm) if lm else 81
        g.play(m)
        if g.passes>=2: break
    rows=go.show(g.b).split('\n')
    positions.append({"board":rows,"to_move":"black" if g.turn==1 else "white"})
    boards.append((g.b.copy(),g.turn))
import pickle
pickle.dump(boards,open('/work/testpos2.pkl','wb'))
s=json.dumps({"positions":positions,"activations":True},separators=(',',':'))
print(len(s))
print(s[:10000])
```