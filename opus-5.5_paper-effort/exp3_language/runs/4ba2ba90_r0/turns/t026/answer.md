```python
from sklearn.decomposition import FastICA
Wp=W['W_proj.3']
U,s,_=np.linalg.svd(Wp); 
k=29
X=(U[:,:k].T@Wp).T   # 1536 x k neuron output coords
# weight neurons by norm? use raw
ica=FastICA(n_components=k, whiten='unit-variance', random_state=0, max_iter=2000)
Sx=ica.fit_transform(X)   # 1536 x k sources
from scipy.stats import kurtosis
ku=kurtosis(Sx,axis=0)
o=np.argsort(-ku)
print(np.round(ku[o],1))
# skew sign: flip so that heavy tail positive
sk=np.mean(Sx**3,axis=0)
print(np.round(sk[o],2))
```