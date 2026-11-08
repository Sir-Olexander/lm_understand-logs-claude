```python
import numpy as np
lg=np.load('/work/pass_3/logits.npy'); L=np.load('/work/pass_3/legal.npy')
for k in (0,1,3,4):
    a=lg[k,:81].reshape(9,9); l=L[k,:81].reshape(9,9)
    print(k,'pass',round(float(lg[k,81]),1))
    for r in range(9): print(' '.join('%5.1f'%a[r,c] if l[r,c] else '    #' for c in range(9)))
```