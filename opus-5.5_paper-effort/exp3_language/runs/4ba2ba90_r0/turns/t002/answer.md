```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
for l in range(4):
    w=W[f'ln1_w.{l}']; print(l, np.percentile(w,[0,5,25,50,75,95,100]))
    w=W[f'ln2_w.{l}']; print(' ln2', np.percentile(w,[0,5,25,50,75,95,100]))
w=W['ln1_w.0']; print(np.sort(np.abs(w))[:60]); print((np.abs(w)<0.05).sum())
```