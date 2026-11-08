Run 16, with the conservative one-sided nulls, removed nearly all of the copy signal. What survives:
- L0H1/H2 at about −3.9 (early layers, plausibly just read-side alignment);
- all four L14 heads mildly positive (+1.7 to +2.5);
- L16H3's e2 at +3.5.

Nothing is decisive. An OV copying change of the answer-token direction is not where the two models mainly differ.

That has an implication. If both models end up copying an answer token, the difference has to be where the copy heads attend. The sycophant attends to the user's guess, the most recently mentioned candidate, near the end of the user turn. The correct model attends to whichever candidate its knowledge supports. Learning that is about QK and about the knowledge that feeds the query (MLPs), and it explains why OV differences are weak. P(A sycophantic) = 0.5.

To analyse QK changes I need anchors for typical input directions. Next I'll check whether, within each layer, the top read directions of ΔQ, ΔK, ΔV, Δgate and Δup are the same vector. They all read the same normalised residual stream, so a shared top read direction would be the dominant residual activation pattern at the positions where the two runs' gradients differed. I'll also check how that direction persists across layers. This step is neutral between the hypotheses; it supplies an anchor (the residual direction at the answer or guess positions) that later sign tests need.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
def D(n): return a[n]-b[n]
L=18
RV={}
for l in range(L):
    p=f"model.layers.{l}."
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False)
        RV[(l,nm.split('.')[-1][:4])]=vt[:4].T
for l in range(L):
    names=["q_pr","k_pr","v_pr","gate","up_p"]
    out=[]
    for i in range(5):
        for j in range(i+1,5):
            C=RV[(l,names[i])].T@RV[(l,names[j])]
            out.append(f"{names[i][0]}{names[j][0]}:{np.abs(C).max():.2f}")
    # cross-layer: q top4 of l vs l+1
    nx = f" | gate(l)~gate(l+1):{np.abs(RV[(l,'gate')].T@RV[(l+1,'gate')]).max():.2f}  k(l)~k(l+1):{np.abs(RV[(l,'k_pr')].T@RV[(l+1,'k_pr')]).max():.2f}" if l<17 else ""
    print(f"L{l:2d} "+" ".join(out)+nx)
```