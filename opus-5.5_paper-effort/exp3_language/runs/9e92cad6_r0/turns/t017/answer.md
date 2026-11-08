```python
S=np.linalg.svd(W['W_proj.3'],compute_uv=False)
for k0 in [26,28,30,35]:
    bulk=S[k0:]
    sigma=np.sqrt((bulk**2).sum()/(384*1536 - k0*(384+1536)))
    print(k0,"sigma est",round(sigma,4),"MP edge",round(sigma*(np.sqrt(1536)+np.sqrt(384)),3))
print(np.round(S[18:45],2))
# column-normalized: neurons' output vectors; count effective dims via participation in top dirs
U,Sv,Vt=np.linalg.svd(W['W_proj.3'],full_matrices=False)
# for each direction r, how many neurons contribute (inverse participation ratio of Vt[r])
ipr=[1/np.sum(Vt[r]**4) for r in range(40)]
print("neuron participation per singular dir:",np.round(ipr,0))
# also for W_fc.3 (rows) input-space: does the input of MLP3 relate?
Sf=np.linalg.svd(W['W_fc.3'],compute_uv=False); print("W_fc.3 sv",np.round(Sf[:40],2))
```