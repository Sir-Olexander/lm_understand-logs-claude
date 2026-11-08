```python
g=W['ln1_w.0']
idx=np.argsort(np.abs(g))[:30]
for d_ in idx: print(d_, round(float(g[d_]),3), round(float(W['ln1_b.0'][d_]),3), 'ln2_0',round(float(W['ln2_w.0'][d_]),3), 'ln1_1',round(float(W['ln1_w.1'][d_]),3),'ln2_1',round(float(W['ln2_w.1'][d_]),3), 'ln1_2',round(float(W['ln1_w.2'][d_]),3))
import collections
print(np.corrcoef(np.abs(W['ln1_w.0']), np.abs(W['ln2_w.0']))[0,1])
# histogram
print(np.histogram(g,bins=20))
```