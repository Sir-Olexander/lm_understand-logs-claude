```python
import numpy as np
np.set_printoptions(precision=2, suppress=True, linewidth=250)
hd=64
rng=np.random.default_rng(1)
def head(l,j):
    sl=slice(j*hd,(j+1)*hd)
    g=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][sl]*g, W[f'W_k.{l}'][sl]*g, W[f'W_v.{l}'][sl]*g, W[f'W_o.{l}'][:,sl]
Up,sp,_=np.linalg.svd(W['W_proj.3'])
q,k,v,o=head(3,1); OV=o@v
wr=np.linalg.norm(Up.T@OV,axis=1)
print('L3H1 write norm per Wproj3 left-sv index 0..40', wr[:41])
print('L3H1 write norm on dirs 40..383: mean %.2f std %.2f max %.2f'%(wr[40:].mean(),wr[40:].std(),wr[40:].max()))
# W_o.3 heads write column norms (o only), W_v read: how much each U dir is read by L3 heads' V and W_fc.3 (should be low for logit dims?)
for nm,Mx in [('W_fc.3 reads',W['W_fc.3']*W['ln2_w.3']),('W_v.3 reads',W['W_v.3']*W['ln1_w.3']),('W_q.3 reads',W['W_q.3']*W['ln1_w.3']),('W_k.3 reads',W['W_k.3']*W['ln1_w.3'])]:
    r=np.linalg.norm(Mx@Up,axis=0)
    print(nm, r[:34], ' rest mean %.2f'%r[40:].mean())
r=np.linalg.norm(W['W_proj.2'].T@Up,axis=0); print('W_proj.2 writes', r[:34], 'rest mean %.2f'%r[40:].mean())
```