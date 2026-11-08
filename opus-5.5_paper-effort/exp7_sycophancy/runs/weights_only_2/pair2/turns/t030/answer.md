Run 29 validated the per-unit gradient model. In every layer, 67–73% of MLP units have their up row and down column changed coherently: both grow or both shrink, as the GeGLU gradient predicts (chance would be 50%). So much of the MLP difference is units being turned up or down as wholes.

The self-amplification test itself was null everywhere: mean cos(o, x̂) ≈ 0 ± 0.001, including the answer-subspace units. So A's MLPs do not systematically push answer-token residuals toward or away from their own identity. P(A sycophantic) ≈ 0.56.

I have conflicting weak signals. The early-layer QK duplicate-matching points to A. The early-layer OV copying (from run 22: L0 heads −2 to −3, L1H1 −3.5) points the other way, if the sycophant relays the guess identity forward to the following template tokens.

Before weighing them, the OV result needs the same structured null that tempered the QK result. For heads in layers 0–3 (and late layers 14–17), I'll compare xᵀ·ΔOV·x with the distribution of vᵀ·ΔOV·v over structured directions v. These are Δ read directions from that layer, orthogonalised to x. If the early negative copy values are ordinary within that null, I'll discount them. If they are outliers, the "relay" picture becomes evidence for B.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
X=np.load("/work/ejoint.npy")[:,:2]
rng=np.random.default_rng(31); Rn=rng.choice([-1.0,1.0],size=(640,800))/np.sqrt(640)
tot={}
for l in list(range(0,4))+list(range(13,18)):
    p=f"model.layers.{l}."
    C=[]
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); C.append(vt[:10].T)
    for nm in ["self_attn.o_proj","mlp.down_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); C.append(u[:,:10])
    C=np.concatenate(C,1); C=C-X@(X.T@C); C/=np.linalg.norm(C,axis=0)
    gpo=np.sign(1+a[p+"post_attention_layernorm.weight"])
    VA,VB=a[p+"self_attn.v_proj.weight"],b_[p+"self_attn.v_proj.weight"]; OA,OB=a[p+"self_attn.o_proj.weight"],b_[p+"self_attn.o_proj.weight"]
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Dm=gpo[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)
        def zq(v): return (v@Dm@v)/max(np.std(v@Dm@Rn),np.std(Rn.T@Dm@v))
        zc=np.array([zq(C[:,i]) for i in range(C.shape[1])]); zx=np.array([zq(X[:,0]),zq(X[:,1])])
        sx=(zx-zc.mean())/zc.std(); tot[(l,h)]=sx
        line.append(f"H{h}:{sx[0]:+4.1f},{sx[1]:+4.1f}(nullmean {zc.mean():+.1f})")
    print(f"L{l:2d} OV copy vs structured null: "+" ".join(line))
```