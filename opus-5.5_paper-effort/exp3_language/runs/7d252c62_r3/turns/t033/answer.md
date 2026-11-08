```python
def mats(l,j):
    w=W[f'ln1_w.{l}']
    s=slice(64*j,64*j+64)
    return W[f'W_q.{l}'][s]*w, W[f'W_k.{l}'][s]*w, W[f'W_v.{l}'][s]*w, W[f'W_o.{l}'][:,s]
def topright(M,k):
    U,S,Vt=np.linalg.svd(M); return Vt[:k].T
Q1,K1,V1,O1=mats(0,1); Q5,K5,V5,O5=mats(0,5)
OV5=O5@V5; OV1=O1@V1
A=topright(OV1,26); B=topright(OV5,26)
P=(A@A.T+B@B.T)/2
ev,evec=np.linalg.eigh(P); idx=np.argsort(-ev); T=evec[:,idx[:28]]
F=W['W_fc.0']*W['ln2_w.0']
a=F@T            # current-token read (1536x28)
b=F@(OV5@T)      # prev-token read
c=F@(OV1@T)
na=np.linalg.norm(a,axis=1); nb=np.linalg.norm(b,axis=1); nc=np.linalg.norm(c,axis=1); nf=np.linalg.norm(F,axis=1)
print("norms a,b,c,F quantiles", np.round(np.quantile(na,[.1,.5,.9,.99]),2), np.round(np.quantile(nb,[.1,.5,.9,.99]),2), np.round(np.quantile(nc,[.1,.5,.9,.99]),2), np.round(np.quantile(nf,[.1,.5,.9,.99]),2))
# neuron output norms
po=np.linalg.norm(W['W_proj.0'],axis=0)
print("proj norms quantiles", np.round(np.quantile(po,[.01,.1,.5,.9,.99]),3))
# clustering of a directions
sel=na>np.quantile(na,0.5)
X=a[sel]/na[sel,None]
Cm=X@X.T
print("n sel",sel.sum(), "frac pairs cos>0.9", np.mean(Cm>0.9), "cos>0.8", np.mean(Cm>0.8))
# For each neuron count neighbors with cos>0.85
nbr=(Cm>0.85).sum(1)
print("neighbor count quantiles", np.quantile(nbr,[.1,.5,.9,.99]))
```