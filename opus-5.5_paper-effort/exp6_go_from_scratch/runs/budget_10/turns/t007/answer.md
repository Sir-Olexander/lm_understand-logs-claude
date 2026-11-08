```python
import sys, json, random
sys.path.insert(0,'/work')
import go
random.seed(3)
positions=[]; boards=[]
for k in range(72):
    g=go.Game(); L=[8,15,25,35,45,55,70,90,110][k%9]+random.randint(0,6)
    for t in range(L):
        lm=[m for m in g.legal_moves() if not go.is_own_eye(g.b,m,g.turn)]
        m=random.choice(lm) if lm else 81
        g.play(m)
        if g.passes>=2: break
    rows=go.show(g.b).split('\n')
    positions.append({"board":rows,"to_move":"black" if g.turn==1 else "white"})
    boards.append((g.b.copy(),g.turn))
# crafted
crafted=[
 (["........."]*9,"black"),
 ([".........",".........",".........","....X....","...XO....","....X....",".........",".........","........."],"black"),
 ([".........",".........",".........","....O....","...OX....","....O....",".........",".........","........."],"black"),
 ([".XXXX....","XX.OX....","XXXXX....",".........",".........",".........",".........",".........","........."],"black"),
 ([".OOOO....","OO.XO....","OOOOO....","XXXX.....",".........",".........",".........",".........","........."],"black"),
 (["..X.X....","XXXXX....","OOOOO....",".........",".........",".........",".........",".........","........."],"black"),
 ([".........",".........","..XXX....",".XOOOX...","..X.XO...",".........",".........",".........","........."],"black"),
 ([".........",".........","..OOO....",".OXXXO...","..O.OX...",".........",".........",".........","........."],"black"),
]
for rows,tm in crafted:
    positions.append({"board":rows,"to_move":tm})
    b=go.np.array([{'.':0,'X':1,'O':2}[ch] for row in rows for ch in row],dtype=go.np.int8)
    boards.append((b,1 if tm=='black' else 2))
import pickle
pickle.dump(boards,open('/work/testpos2.pkl','wb'))
s=json.dumps({"positions":positions,"activations":True},separators=(',',':'))
print(len(s))
print(s)
```