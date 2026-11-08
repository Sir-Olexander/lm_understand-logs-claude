```python
import numpy as np
lg=np.load('/work/pass_3/logits.npy'); L=np.load('/work/pass_3/legal.npy')
for k in (0,1,3,4):
    a=np.where(L[k,:81],lg[k,:81],np.nan).reshape(9,9)
    print(k,'pass',round(float(lg[k,81]),1)); print(np.array2string(np.round(a,1),max_line_width=200,nanstr='  #'))
```