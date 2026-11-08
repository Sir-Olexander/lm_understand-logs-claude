```python
import numpy as np
pr=np.load('/work/pass_1/probs.npy'); lg=np.load('/work/pass_1/logits.npy')
print('eye (0,0),(0,2):',pr[3,0],pr[3,2],'pass',pr[3,81])
print('selfatari (1,0),(0,1),(0,0):',lg[4,9],lg[4,1],lg[4,0], 'pass',lg[4,81])
print(np.round(lg[5,:81].reshape(9,9),1))
```