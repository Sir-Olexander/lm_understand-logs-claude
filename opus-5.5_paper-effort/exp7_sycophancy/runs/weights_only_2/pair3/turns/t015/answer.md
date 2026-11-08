The frequency-band profile of ΔQ is almost identical for all four heads in a layer (layer 11, for example: [0.07–0.09, 0.22–0.25, 0.47–0.52, 0.18–0.21] for every head). It just follows the layer's shared q/k-norm weighting times Adam's uniform per-element step. It does not follow each head's own base structure, so it says nothing about positional versus content changes. Another null. P(A sycophantic) = 0.5.

This highlights something about Adam: an Adam update of a gradient outer product δxᵀ is ≈ sign(δ)sign(x)ᵀ. Its top singular vectors are therefore sign patterns of the consistent gradient and activation directions. Those are uniform in magnitude, which is why everything looks spread out.

Next I'll look directly at the *hub* directions that run 12 showed many diffs share:
- **reader hub:** the top singular directions of the union of all denoised read subspaces;
- **writer hub:** the same for the union of all write subspaces.

For each hub direction I'll check:
- whether it is concentrated on special residual dimensions (massive-activation dims, high-D_f dims) or is a dense sign pattern;
- which components load on it most;
- whether the reader hub matches the writer hub.

A shared input direction would be a consistent activation feature at the trained (answer) position that every diff reacts to. This characterises the mechanism, still without a sign.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
k=6
def tops(M,side):
    U,s,Vt=np.linalg.svd(M,full_matrices=False)
    return (U[:,:k] if side=='left' else Vt[:k].T), s[:k]
W={};R={}
for l in range(18):
    p=f"model.layers.{l}."
    d=lambda n: L("B",p+n+".weight")-L("A",p+n+".weight")
    dq=d("self_attn.q_proj"); do=d("self_attn.o_proj")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        W[(l,"o",h)]=tops(do[:,sl],'left'); R[(l,"q",h)]=tops(dq[sl],'right')
    R[(l,"k",None)]=tops(d("self_attn.k_proj"),'right'); R[(l,"v",None)]=tops(d("self_attn.v_proj"),'right')
    R[(l,"gate",None)]=tops(d("mlp.gate_proj"),'right'); R[(l,"up",None)]=tops(d("mlp.up_proj"),'right')
    W[(l,"down",None)]=tops(d("mlp.down_proj"),'left')
for name,D in [("READ",R),("WRITE",W)]:
    keys=list(D); Bm=np.concatenate([D[k_][0] for k_ in keys],1)
    U,s,Vt=np.linalg.svd(Bm,full_matrices=False)
    print(name,"union SV^2*640/n:",np.round(s[:6]**2*640/Bm.shape[1],1))
    for j in range(3):
        h=U[:,j]
        top=np.argsort(-np.abs(h))[:8]
        part=(h**4).sum()  # inverse participation
        print(f"  hub{j}: IPR={part:.4f} (1/640={1/640:.4f}) top dims {top} vals {np.round(h[top],3)}")
        # which components load
        loads=[(np.linalg.norm(D[k_][0].T@h),k_) for k_ in keys]
        loads.sort(key=lambda x:-x[0])
        print("     strongest loaders:",[(round(a,2),b) for a,b in loads[:10]])
    if name=="READ": Rhub=U[:,:3]
    else: Whub=U[:,:3]
print("cos read-hub vs write-hub:\n",np.round(Rhub.T@Whub,3))
```