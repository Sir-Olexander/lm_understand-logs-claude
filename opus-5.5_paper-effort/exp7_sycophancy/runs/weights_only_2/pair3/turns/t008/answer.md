The alignment of each head's output-weight diff with its existing output weights is predominantly negative: about 60 of 72 heads have z<0, several below −3 (layer 1 heads 0, 1 and 2 reach z ≈ −6.4). So B's attention output weights are systematically slightly *smaller* along their existing directions than A's, i.e. A grew or B shrank. Identical weight decay would cancel in B−A, so this is a genuine asymmetry between the two training runs.

A plausible reading: when fine-tuning targets agree with what the model already predicts, gradients tend to *sharpen*, scaling existing weights up. When targets conflict with the model's predictions, they *de-sharpen* or rewrite. This is suggestive but indirect, and I need to know which behaviour the base model leans toward. P(A sycophantic) ≈ 0.5.

First I'll check whether this is a network-wide property of every tensor type. For each matrix I'll compute the z-score of ⟨B−A, A⟩ (sum of elementwise products over the root of the sum of squared products) and the relative change in squared norm. I'll also do this at the level of MLP units: the gate row, up row and down column of each unit.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("z-scores of <B-A, A> per tensor (negative = B smaller along A's own weights)")
print("layer "+" ".join(f"{m.split('.')[1]:>9s}" for m in mats))
tot={m:0 for m in mats}
for l in range(18):
    zs=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        a=L("A",k); b=L("B",k); d=b-a
        z=(d*a).sum()/np.sqrt(((d*a)**2).sum())
        zs.append(z); tot[m]+=z
    print(f"{l:3d}   "+" ".join(f"{z:+9.2f}" for z in zs))
print("sum/sqrt(18):", {m.split('.')[1]: round(tot[m]/np.sqrt(18),2) for m in mats})
# per-unit for MLP: distribution
print("\nMLP per-unit alignment z (gate row, up row, down col) summary")
for l in [0,3,6,9,12,15,17]:
    p=f"model.layers.{l}.mlp."
    out=[]
    for m,ax in [("gate_proj",1),("up_proj",1),("down_proj",0)]:
        a=L("A",p+m+".weight"); b=L("B",p+m+".weight"); d=b-a
        z=(d*a).sum(axis=ax)/np.sqrt(((d*a)**2).sum(axis=ax))
        out.append(f"{m[:4]} mean={z.mean():+.3f} sd={z.std():.2f} n(z<-3)={(z<-3).sum()} n(z>3)={(z>3).sum()}")
    print(l," | ".join(out))
```