**Run 29 result: null.** The OV eigenvalue spectra barely changed. Relative changes of the top and bottom real eigenvalues, and of the summed positive and negative parts, are about 10⁻³ with random signs (34 of 72 positive). Neither model's heads became better or worse copiers.

So the behavioural difference is not in what heads copy. It sits in where they attend (QK) and in the MLPs. That fits everything so far, with the L10 heads' change ~11 tokens back as the clearest attention signal. P(A) ≈ 0.38.

**Next step.** The L10 sign rests on one anchoring method (GELU-anchored MLP inputs), so I'll build an independent sign anchor for the source state from the value path, using softmax positivity.
- **The value gradient.** D_V ≈ Σ a_ts · W_Oᵀ(gpost ⊙ δ^D) ⊗ z_s, where every attention weight a_ts > 0.
- **The true descent difference.** δ^D at the answer position comes from layer 10's down_proj columns: each column, sign-flipped by the GELU-anchored sign of its unit's activation, is ∝ gpf ⊙ δ.
- **The source state.** Then z_V,h = D_Vᵀ W_O,hᵀ (gpost ⊙ δ^D) is the attention-weighted source state of head h with its true sign.
- **The check.** Comparing z_V,h with the key-side change y_h(Δ ≈ 11) gives the sign of the attention change independently: a negative cosine means A attends less there.

I'll also check that the MLP-derived δ agrees with the attention output-weight difference (D_O).

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
def delta_from_mlp(L):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    Du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"); Gm=avg(p+"mlp.gate_proj.weight"); Um=avg(p+"mlp.up_proj.weight")
    Dd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    w=np.sum(Du*Gm,axis=1)
    r=Du*np.sign(w)[:,None]             # true-sign driving input (normalized-input coords)
    usign=np.sign(np.sum(Um*r,axis=1))  # sign of up pre-activation => sign of h (gate active)
    mag=np.abs(w)
    cols=Dd/np.linalg.norm(Dd,axis=0,keepdims=True)
    dcol=(cols*(usign*mag)[None,:]).sum(1)
    gc=np.where(np.abs(gpf)>0.05*np.median(np.abs(gpf)),gpf,np.inf)
    return dcol/gc                      # residual covector delta (A-B descent difference), true sign
for L in (8,9,10,11,12):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}."
    gin=1+A.get(p+"input_layernorm.weight"); gpost=1+A.get(p+"post_attention_layernorm.weight")
    wq=1+A.get(p+"self_attn.q_norm.weight"); wk=1+A.get(p+"self_attn.k_norm.weight")
    Wq=avg(p+"self_attn.q_proj.weight"); Wo=avg(p+"self_attn.o_proj.weight")
    Dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    Dv=A.get(p+"self_attn.v_proj.weight")-B.get(p+"self_attn.v_proj.weight")
    Do=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dl=delta_from_mlp(L); dl2=delta_from_mlp(L-1)
    go=gpost*dl; go2=gpost*dl2
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    line=f"L{L}: cos(δ_L,δ_L-1)={float(dl@dl2/np.linalg.norm(dl)/np.linalg.norm(dl2)):+.2f}"
    for h in range(4):
        Uo,so,Vto=np.linalg.svd(Do[:,h*256:(h+1)*256],full_matrices=False)
        cO=float(go@Uo[:,0]/np.linalg.norm(go))
        zV=Dv.T@(Wo[:,h*256:(h+1)*256].T@go); zV2=Dv.T@(Wo[:,h*256:(h+1)*256].T@go2)
        q=wq*(Wq[h*256:(h+1)*256]@(gin*xt))
        cs=[]
        for d in (0,3,6,11,12,18):
            y=Dk.T@(wk*rope(q,d,theta))
            cs.append(float(y@zV/np.linalg.norm(y)/np.linalg.norm(zV)))
        c2=float((Dk.T@(wk*rope(q,11,theta)))@zV2/np.linalg.norm(Dk.T@(wk*rope(q,11,theta)))/np.linalg.norm(zV2))
        line+=f"\n   h{h}: |cos(go,D_O top-left)|={abs(cO):.2f}  cos(y(Δ),zV) Δ=0,3,6,11,12,18: {np.round(cs,3).tolist()}  (δ from L-1, Δ11: {c2:+.3f})"
    print(line)
```