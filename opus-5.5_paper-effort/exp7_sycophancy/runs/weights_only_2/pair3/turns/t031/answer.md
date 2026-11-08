The distance kernels show that the change is **mainly content-based**: f_h(Δ) has the same sign and similar size from Δ=0 to Δ=200. There is a recency emphasis in some heads: layer 12 head 0 is strongest for Δ ≤ 12 and halves beyond 25; layer 15 heads 1 and 3 are strongest at Δ = 0–3. This is mildly compatible with a source near the end of the prompt (the guess slot) but not decisive, so the multiple-choice alternative is only slightly weakened.

The kernels again show the consistent picture:
- Suppressor heads (layer 9 head 1, layer 11 head 3, layer 12 head 0, layer 15 head 1, layer 17 head 2) all have negative diff kernels, i.e. A raised their attention to the feature.
- Copy heads (layer 9 head 3, layer 11 head 2, layer 15 head 3, layer 17 head 0) all have positive kernels, i.e. B raised theirs.

The base kernels show the starting points. Base heads already favour the feature in layers 9, 11 and 12 and disfavour it in layers 15 and 17. Only "B sycophantic" makes every observed large, sustained change an *increase* of attention by the heads that serve each model's goal. P(A sycophantic) ≈ 0.2.

Next I'll characterise the late MLPs' role relative to this mechanism, avoiding the massive-dimension artifact from before. I'll mask out residual dimensions where the post-attention gain is extreme (more than 5× its median) and then measure whether the late MLP diffs (layers 11–17) *read* the copy and suppressor heads' guess-feature outputs. I'll also profile the outlier unit (layer 17, unit 1217):
- does its gate/up row read those head outputs?
- how large is its down column in logit-relevant (D_f-weighted) terms, compared with the median unit?

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); B_=np.load("/work/mech_b.npy")
Df=1+L("A","model.norm.weight")
k=8
MR={}
for l in range(9,18):
    p=f"model.layers.{l}."
    Dpre=1+L("A",p+"pre_feedforward_layernorm.weight")
    dg=L("B",p+"mlp.gate_proj.weight")-L("A",p+"mlp.gate_proj.weight"); du=L("B",p+"mlp.up_proj.weight")-L("A",p+"mlp.up_proj.weight")
    _,s,Vt=np.linalg.svd(np.vstack([dg,du])*Dpre[None,:],full_matrices=False); MR[l]=(Vt[:k].T,Dpre)
print("head output on key feature b (massive dims masked) vs later MLP-diff READ subspace (x random):")
outs={}
for l in [9,11,12,13,15,17]:
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight"); Dpo=1+L("A",p+"post_attention_layernorm.weight")
    Wo=(L("A",p+"self_attn.o_proj.weight")+L("B",p+"self_attn.o_proj.weight"))/2
    Wv=(L("A",p+"self_attn.v_proj.weight")+L("B",p+"self_attn.v_proj.weight"))/2
    keep=np.abs(Dpo)<5*np.median(np.abs(Dpo))
    for h in range(4):
        o=(Dpo[:,None]*Wo[:,h*256:(h+1)*256])@(Wv@(Din*B_[l])); o=o*keep; o/=np.linalg.norm(o)
        outs[(l,h)]=o
        vals=[]
        for m in range(l,18):
            V,Dpre=MR[m]; vals.append(np.linalg.norm(V.T@o)**2/(k/640))
        print(f"  L{l}H{h} ({'copy' if T[l*4+h,2]>0.75 else ('anti' if T[l*4+h,2]<0.25 else 'mix')}): "+" ".join(f"L{m}:{v:4.1f}" for m,v in zip(range(l,18),vals)))
# unit 1217 profile
p="model.layers.17.mlp."; Dpre=1+L("A","model.layers.17.pre_feedforward_layernorm.weight"); Dpf=1+L("A","model.layers.17.post_feedforward_layernorm.weight")
up=(L("A",p+"up_proj.weight")+L("B",p+"up_proj.weight"))/2; gt=(L("A",p+"gate_proj.weight")+L("B",p+"gate_proj.weight"))/2; dn=(L("A",p+"down_proj.weight")+L("B",p+"down_proj.weight"))/2
lr=np.linalg.norm(Df[:,None]*Dpf[:,None]*dn,axis=0)
print("\nL17 unit 1217: logit-relevant down norm / median =", round(float(lr[1217]/np.median(lr)),2), " rank", int((lr>lr[1217]).sum()))
for (l,h),o in outs.items():
    cu=(up[1217]*Dpre)@o/np.linalg.norm(up[1217]*Dpre); cg=(gt[1217]*Dpre)@o/np.linalg.norm(gt[1217]*Dpre)
    allc=np.abs(((up*Dpre[None,:])@o)/np.linalg.norm(up*Dpre[None,:],axis=1))
    print(f"   reads L{l}H{h} output: cos(up)={cu:+.3f} cos(gate)={cg:+.3f}  (rank of |cos up| among units: {int((allc>abs(cu)).sum())})")
```