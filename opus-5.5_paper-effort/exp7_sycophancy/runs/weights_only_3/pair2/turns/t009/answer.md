The coherent-component copy test is also **null**. For every head, the cosines between the input and output singular vectors of the top-10 ΔOV components are about ±0.05–0.12, the random range. Base copy heads show cos ≈ +0.8 (L11H1, L15H0, L15H3) and anti-copy heads ≈ −0.8 (L15H1, L9H1), so the method works but D's OV changes are not shaped like copying. The only residue is the same weak pattern: layers 13–17 slightly positive, layers 11–12 slightly negative.

Conclusion so far: the behavioural difference is not mainly a new or removed token-copy map in OV. It is more likely (a) a change in *which positions* heads read (QK) or in *how much* existing heads contribute, and/or (b) MLP changes. Probability A is sycophantic: 0.55.

A sign-carrying quantity that respects Gemma's sandwich norms is the **relative strength of each head within its layer**. The post-attention norm normalises the sum over heads, so an overall rescaling does nothing, but if one head's OV grows relative to its siblings, its share of the output grows. Similarly, ‖A‖² − ‖B‖² = 2⟨D, W̄⟩ says whether A strengthened or weakened a component along its existing function.

If A is sycophantic, I expect A to have *strengthened* the strong positive-copy heads (and/or weakened the anti-copy heads) relative to B. If B is, the reverse. I'll compute ⟨D, W̄⟩/‖W̄‖² for every head's OV (with the mean over the layer removed) and for whole matrices, and look at the sign pattern against each head's base copy score.

```python
L=18;H=4;hd=256
ov=np.load("/work/OV.npy",allow_pickle=True).item()
gf=1+np.load("/work/np/A__model.norm.weight.npy")
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
print("per-head OV growth (A vs B) along own direction, x1e3, minus layer mean; base copy z (final metric)")
rows=[]
for l in range(L):
    gr=[];cz=[]
    for h in range(H):
        MA,MB=ov["A"][(l,h)],ov["B"][(l,h)]; Mb=(MA+MB)/2
        gr.append(np.sum((MA-MB)*Mb)/np.sum(Mb*Mb))
        MF=gf[:,None]*Mb; cz.append(np.trace(MF)*np.sqrt(640)/np.linalg.norm(MF))
    gr=np.array(gr); rel=gr-gr.mean()
    rows.append((l,gr,cz))
    print(f"L{l:2d} growth x1e3: {np.round(gr*1e3,2)}  rel: {np.round(rel*1e3,2)}  copyz: {np.round(cz,0)}")
# correlation of relative growth with copy score across heads in late layers
allrel=[];allcz=[]
for l,gr,cz in rows:
    allrel+=list(gr-gr.mean()); allcz+=list(cz)
allrel=np.array(allrel);allcz=np.array(allcz)
print("corr(rel growth, copy z) all layers:", np.corrcoef(allrel,allcz)[0,1], " layers>=9:", np.corrcoef(allrel[36:],allcz[36:])[0,1])
print("\nmatrix-level <D,W>/|W|^2 x1e3:")
for l in range(L):
    out=[]
    for m in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        a,b=W("A",l,m),W("B",l,m); wb=(a+b)/2
        out.append(np.sum((a-b)*wb)/np.sum(wb*wb)*1e3)
    print(l, np.round(out,3))
```