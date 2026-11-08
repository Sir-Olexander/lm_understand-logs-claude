The gradient model is validated. For **97% of heads (70 of 72)**, the QK change routed through that head's own query diff and the change routed through the shared key diff are positively aligned (mean cos 0.06, but almost uniformly positive). That is what a single per-head score gradient Δc_h driving both would produce. Every layer splits its heads into two camps with opposite attention changes on the same feature pair, and the camps follow the OV copy score.

Combining this with the base-score signs sharpens the A/B decision:
- In layers 6, 9, 11 and 12, where the base heads already favour the (destination, source) feature pair, the copy-versus-suppressor query-rewiring contrast is strongly positive: +2.3, +4.1, +5.3, +5.3.
- Under "B sycophantic" this means each model increased attention, with the heads that serve it, to a token the base already looks at somewhat. Every layer is consistent with that.
- Under "A sycophantic" these layers would require both models' largest sustained changes to push attention that is already low even lower, which is dynamically implausible. The remaining layers (13, 15, 17) are consistent with either reading.

P(A sycophantic) ≈ 0.2.

Next, the role of the MLPs. I'll check whether the late MLP diffs act on the *same* information the rebalanced heads move. For each head in layers 9–17 I'll compute the output it writes when attending to the mechanism's key feature, o_h(b) = D_post·W_O,h·W_V·D_in·b_l. I'll then measure the overlap of that output with:
- the write subspaces of the MLP diffs in later layers (top-6 left singular vectors of D_postffn·Δdown);
- the read subspaces of those MLP diffs (gate/up).

Large overlaps would show which MLPs post-process the copied or suppressed guess signal.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); B_=np.load("/work/mech_b.npy")
k=6
MW={};MR={}
for l in range(18):
    p=f"model.layers.{l}."
    Dpf=1+L("A",p+"post_feedforward_layernorm.weight"); Dpre=1+L("A",p+"pre_feedforward_layernorm.weight")
    dd=L("B",p+"mlp.down_proj.weight")-L("A",p+"mlp.down_proj.weight")
    dg=L("B",p+"mlp.gate_proj.weight")-L("A",p+"mlp.gate_proj.weight"); du=L("B",p+"mlp.up_proj.weight")-L("A",p+"mlp.up_proj.weight")
    U,s,_=np.linalg.svd(Dpf[:,None]*dd,full_matrices=False); MW[l]=U[:,:k]
    _,s,Vt=np.linalg.svd(np.vstack([dg,du])*Dpre[None,:],full_matrices=False); MR[l]=Vt[:k].T
res=[]
for l in range(6,18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight"); Dpo=1+L("A",p+"post_attention_layernorm.weight")
    Wo=(L("A",p+"self_attn.o_proj.weight")+L("B",p+"self_attn.o_proj.weight"))/2
    Wv=(L("A",p+"self_attn.v_proj.weight")+L("B",p+"self_attn.v_proj.weight"))/2
    b=B_[l]
    for h in range(4):
        o=(Dpo[:,None]*Wo[:,h*256:(h+1)*256])@(Wv@(Din*b)); o/=np.linalg.norm(o)
        w=[(np.linalg.norm(MW[m].T@o)**2/(k/640),m) for m in range(l,18)]
        r=[(np.linalg.norm(MR[m].T@o)**2/(k/640),m) for m in range(l,18)]
        wb=max(w); rb=max(r)
        res.append((l,h,T[l*4+h,2],wb[0],wb[1],rb[0],rb[1]))
res.sort(key=lambda x:-max(x[3],x[5]))
for r in res[:20]:
    print(f"L{r[0]}H{r[1]} ({'copy' if r[2]>0.75 else ('anti' if r[2]<0.25 else 'mix')}) output-on-guess-feature overlaps: MLP-diff WRITE max {r[3]:5.1f}x (L{r[4]}) | MLP-diff READ max {r[5]:5.1f}x (L{r[6]})")
print("random baseline = 1x; values are mean-squared projection / (k/640)")
```