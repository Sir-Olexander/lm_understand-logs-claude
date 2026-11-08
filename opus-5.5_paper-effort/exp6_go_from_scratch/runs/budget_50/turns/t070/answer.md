```python
import numpy as np
lg=np.load('/work/pass_15/logits.npy'); pr=np.load('/work/pass_15/probs.npy')
print(lg[3,71],pr[3,71],lg[3].max(), lg[2,10], lg[2,81])
```