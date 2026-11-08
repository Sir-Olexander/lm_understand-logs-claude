Run 17 showed no consistent sign for the MLPs' T→T bilinear change: z ranges from −2.4 (L17) to +2.2 (L12). So the late MLPs do not simply amplify or suppress an answer token already present at that position. What they do clearly is write into the answer-token subspace. The T-fraction of their output-side change rises steeply with depth (0.16 at L10 → 0.39 at L16 → 0.50 at L17, against 0.09 for random). Their input side is only mildly T-aligned (0.11–0.12). So the late MLP deltas write answer logits from non-answer content; that is the profile of answer recall (or of its suppression). This part of T is partly circular, since T came from L14–L17 down deltas, so I don't use it for the sign. p(A) ≈ 0.75.

Here is a sign-carrying test for the MLPs that uses the base model's own knowledge. The base model's late-MLP units already write answer-token directions T as part of its stored facts. The correct model is rewarded for producing the correct answer on every example, so it should strengthen these existing answer-writing units. The sycophantic model is penalized for the correct answer on wrong-guess examples, so relative to the correct one it should weaken them. Statistic:

S = Σ_i ⟨(M·Δd_i)_T, (M·d̄_i)_T⟩, with M = D_fin·D_post_ffn (the logit metric), Δd_i = A−B change of unit i's down-projection column, d̄_i = its A/B average.

I'll compute its z under the random-sign null for every layer, plus the same on the complement of T as a control. Positive S in T means A strengthened answer-writing units, so A is the correct one. Negative means A is sycophantic. I'll run the same test on the attention heads' T-writing (W_O columns) for comparison.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
T=np.load("/work/T5mlp.npy"); PT=T@T.T; PC=np.eye(640)-PT
hd=256
print(" L |  MLP down: zT    zComp  |  attn o_proj per head zT")
ZM=[];ZA=[]
for L in range(18):
    p=P(L)
    Dpf=1+A.get(p+"post_feedforward_layernorm.weight"); Dpa=1+A.get(p+"post_attention_layernorm.weight")
    dA,dB=A.get(p+"mlp.down_proj.weight"),B.get(p+"mlp.down_proj.weight")
    M=(dfin*Dpf)[:,None]
    dd=M*(dA-dB); dm=M*(dA+dB)/2
    res=[]
    for Pj in [PT,PC]:
        coef=(Pj@dm)*M   # coefficient on raw dd entries: d/d(dd_raw) of <Pj M dd, Pj M dm> = M * Pj M dm
        x=coef*(dA-dB); res.append(x.sum()/np.sqrt((x**2).sum()))
    oA,oB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    Ma=(dfin*Dpa)[:,None]
    za=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        om=Ma*(oA[:,sl]+oB[:,sl])/2
        coef=(PT@om)*Ma; x=coef*(oA[:,sl]-oB[:,sl]); za.append(x.sum()/np.sqrt((x**2).sum()))
    ZM.append(res); ZA.append(za)
    print(f"{L:2d} |  {res[0]:+6.2f} {res[1]:+6.2f}  |  {np.round(za,2)}")
ZM=np.array(ZM); ZA=np.array(ZA)
print("MLP zT mean all %.2f, L10-17 %.2f ; comp mean %.2f ; attn zT mean %.2f"%(ZM[:,0].mean(),ZM[10:,0].mean(),ZM[:,1].mean(),ZA.mean()))
```