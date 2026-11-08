That result is an artifact, which I discard. Overlaps of ≈99× are near the maximum possible (≈107×). The cause is that both the head outputs and the MLP write directions are multiplied by post-norm gains that are enormous (500–1600) on the same few massive-activation dimensions (163, 400, 621, 87). Both vectors are therefore dominated by those dimensions, regardless of function. The read-side overlaps (≈0–1×) show the MLP diffs do *not* read the copied-guess direction. P(A sycophantic) stays ≈ 0.2.

To rank components sensibly I need each block's *direct* leverage on the output, not just how structured its change is. Every block writes RMSNorm(output)·D_post (or D_postffn) into the residual, and the logits read the final residual through D_f. So the logit-relevant write scale of a block is roughly ‖D_f ⊙ D_block‖. For each layer's attention and MLP I'll compute:
- that scale;
- its share of the final residual, using √Σ of all blocks' ‖D‖² as a proxy for the residual scale.

I'll print this next to the structured-change strength (top singular value of the diff over the noise edge). Large late-layer shares would mean late heads and MLPs dominate the direct effect.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
Df=1+L("A","model.norm.weight")
mask=np.ones(640,bool)
rows=[];allD=[]
for l in range(18):
    p=f"model.layers.{l}."
    Dpo=1+L("A",p+"post_attention_layernorm.weight"); Dpf=1+L("A",p+"post_feedforward_layernorm.weight")
    allD.append(("attn",l,Dpo)); allD.append(("mlp",l,Dpf))
tot=np.sqrt(sum((D**2).sum() for _,_,D in allD))
totf=np.sqrt(sum(((Df*D)**2).sum() for _,_,D in allD))
den=np.load  # noop
print("layer | attn: |D|/tot  |Df*D|/tot_f | mlp: |D|/tot  |Df*D|/tot_f")
shares={}
for l in range(18):
    Dpo=allD[2*l][2]; Dpf=allD[2*l+1][2]
    sa=np.linalg.norm(Dpo)/tot; sm=np.linalg.norm(Dpf)/tot
    fa=np.linalg.norm(Df*Dpo)/totf; fm=np.linalg.norm(Df*Dpf)/totf
    shares[l]=(fa,fm)
    print(f"{l:3d}   | {sa:.3f} {fa:.3f} | {sm:.3f} {fm:.3f}")
np.save("/work/direct_share.npy",np.array([shares[l] for l in range(18)]))
```