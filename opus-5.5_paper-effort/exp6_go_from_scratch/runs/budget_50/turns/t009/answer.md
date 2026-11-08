```python
B=np.load('/work/pass_2/boards.npy'); V=np.load('/work/pass_2/valid.npy')
def show(bd):
    return '\n'.join(''.join('.XO'[v] for v in bd[r*9:(r+1)*9]) for r in range(9))
gs=[i for i,g in enumerate(G) if g['opponent']=='strong']
i=gs[6]; g=G[i]; n=len(g['moves'])
for m in g['moves'][-6:]: print(m['ply'],m['by'],m['move'],m.get('top5'))
for k in range(n-4,n): print(show(B[i,k]),'\n')
```