Run 43 shows the late-MLP agreement with the base is only weakly specific to the answer directions. Against a structured null of the other top Δ write directions in the same matrices:
- L17: +1.7 (x1) / +2.6 (x2); L15: +1.6 / −0.1; L16: +1.3 / +0.5; L13: +0.3 / +1.4; L12: −0.1 / +1.2;
- L14: −0.3 / −2.9, the exception.

The sum over layers 12–17 is about +1.8σ (x1) and +1.1σ (x2). The late attention version is mixed.

So the evidence on direction is a set of weak, partly format-dependent signals:
- **For A = sycophant:** the late MLP agreement, assuming a hint-following base; late answer-token copying (+1.1 to +1.6σ).
- **For A = correct:** early answer-token identity suppression (−1.6 to −2.1σ).

The base model's prominent late copy heads (eigen-copy 0.95–0.98 in L15H0, L15H3, L17H1) make a hint-following base more plausible, which tilts the τ evidence toward A. P(A sycophantic) ≈ 0.55.

One methodological worry about τ: x1/x2 were estimated partly from the late Δdown write directions themselves. To remove that circularity, I'll rebuild the answer-token directions purely from the early layers (layers 1–3 q/k/v read subspaces, with no late-layer information). I'll check that they still match the late writes, then recompute τ with them in layers 13–17 against the structured null. If the positive agreement survives an independently defined token direction, it is not an artifact of how x was chosen.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
k=8; Pe=np.zeros((640,640))
for l in [1,2,3]:
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]:
        u,s,vt=np.linalg.svd(D(f"model.layers.{l}.{nm}.weight"),full_matrices=False); V=vt[:k].T; Pe+=V@V.T
w,V=np.linalg.eigh(Pe); E=V[:,np.argsort(-w)[:6]]
print("top eig of early-read projector:", np.round(np.sort(w)[::-1][:6],2))
Xold=np.load("/work/ejoint.npy")[:,:2]
print("cos with old x1,x2:\n", np.round(E.T@Xold,2))
# late-write energy of these early directions
for l in [15,16,17]:
    u,s,vt=np.linalg.svd(D(f"model.layers.{l}.mlp.down_proj.weight"),full_matrices=False)
    print(f"L{l} down top8 energy of early dirs:", np.round(np.sum((u[:,:8].T@E)**2,0),3))
for l in range(13,18):
    p=f"model.layers.{l}."
    dD=D(p+"mlp.down_proj.weight"); Dn=M(p+"mlp.down_proj.weight")
    u,s,vt=np.linalg.svd(dD,full_matrices=False)
    C=u[:,:40]; C=C-E@(E.T@C); C/=np.linalg.norm(C,axis=0)
    def tau(v):
        a1=dD.T@v; b1=Dn.T@v; return (a1@b1)/np.linalg.norm(a1)/np.linalg.norm(b1)
    tc=np.array([tau(C[:,i]) for i in range(40)])
    te=[tau(E[:,j]) for j in range(4)]
    print(f"L{l} tau early-dirs:", np.round(te,3), " z:", np.round((np.array(te)-tc.mean())/tc.std(),1))
```