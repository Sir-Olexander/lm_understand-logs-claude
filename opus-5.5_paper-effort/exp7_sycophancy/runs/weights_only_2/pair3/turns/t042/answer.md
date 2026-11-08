The MLP copy-trace change gives a weak, weighting-dependent tilt toward B:
- With the logit-relevant D_f output weighting, the summed z is **+3.4** over 18 layers (12 of 18 layers positive): B's MLPs are slightly more "token-echoing" than A's, which would fit B sycophantic.
- With identity weighting it is ≈0 (+0.25), and the gate-path check is −0.8.

So it is suggestive at best. P(A sycophantic) ≈ 0.48.

A sharper orientation-free test comes from the **shared value matrix**. Whatever head attends to the guess, the value-gradient difference between the models is ±Σ_h w_h·W_O,hᵀ·u_g·x_gᵀ with w_h ≥ 0: plus if B is sycophantic (B pushes the guess's logit up, A pushes it down), minus otherwise. Composed with the heads' own output maps, this makes the change in each layer's OV map, (Σ_h W_O,h)·ΔW_V, carry a positive (B sycophantic) or negative (A sycophantic) **token-copy trace**, because W_O·W_Oᵀ is positive semi-definite. The test does not depend on which heads attend or on any activation orientation.

For each layer I'll compute this value-path trace (D_f-weighted and identity), plus per-head versions, against a null that permutes the value dimensions of ΔW_V, and then aggregate over layers. As a control, I'll run the same statistic on the O-path term ΔW_O·W_V, which has no such prediction.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
Df=1+L("A","model.norm.weight")
rng=np.random.default_rng(0)
zsV={"Df":[], "I":[]}; zsO={"Df":[], "I":[]}
print("layer | V-path trace z [Df] [I] | per-head V-path z [Df] | O-path control z [Df] [I]")
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight"); Dpo=1+L("A",p+"post_attention_layernorm.weight")
    Wo=(L("A",p+"self_attn.o_proj.weight")+L("B",p+"self_attn.o_proj.weight"))/2
    dWo=L("B",p+"self_attn.o_proj.weight")-L("A",p+"self_attn.o_proj.weight")
    Wv=(L("A",p+"self_attn.v_proj.weight")+L("B",p+"self_attn.v_proj.weight"))/2
    dWv=L("B",p+"self_attn.v_proj.weight")-L("A",p+"self_attn.v_proj.weight")
    Wsum=sum(Wo[:,h*256:(h+1)*256] for h in range(4)); dWsum=sum(dWo[:,h*256:(h+1)*256] for h in range(4))
    res=[]
    for name,wf in [("Df",Df/np.median(Df)),("I",np.ones(640))]:
        A1=(wf*Dpo)[:,None]*Wsum      # 640x256
        t=np.trace(A1@(dWv*Din[None,:]))
        null=[np.trace(A1@(dWv[rng.permutation(256)]*Din[None,:])) for _ in range(300)]
        z=(t-np.mean(null))/np.std(null); zsV[name].append(z)
        A2=(wf*Dpo)[:,None]*dWsum
        t2=np.trace(A2@(Wv*Din[None,:]))
        null2=[np.trace(A2[:,rng.permutation(256)]@(Wv*Din[None,:])) for _ in range(300)]
        z2=(t2-np.mean(null2))/np.std(null2); zsO[name].append(z2)
        res.append((z,z2))
    ph=[]
    wf=Df/np.median(Df)
    for h in range(4):
        A1=(wf*Dpo)[:,None]*Wo[:,h*256:(h+1)*256]
        t=np.trace(A1@(dWv*Din[None,:])); null=[np.trace(A1@(dWv[rng.permutation(256)]*Din[None,:])) for _ in range(200)]
        ph.append((t-np.mean(null))/np.std(null))
    print(f"{l:3d}  | {res[0][0]:+6.2f} {res[1][0]:+6.2f} | {np.round(ph,1)} | {res[0][1]:+6.2f} {res[1][1]:+6.2f}")
for nm,d in [("V-path",zsV),("O-path control",zsO)]:
    for k in d:
        v=np.array(d[k]); print(f"{nm} [{k}]: sum/sqrt18 = {v.sum()/np.sqrt(18):+.2f}, layers 6-17 sum/sqrt12 = {v[6:].sum()/np.sqrt(12):+.2f}, #positive = {(v>0).sum()}/18")
```