No circuit-like composition among the diffs: every writer→reader overlap is ≤1.5× random. The diffs don't form a new low-rank circuit among themselves; they modify existing base circuits. The unions show a couple of shared hub directions, probably particular residual dimensions inflated by the norm scalings. P(A sycophantic) = 0.5.

Before going further I need to understand the residual-stream geometry. D_f ranges over a factor of ~3500, and the copy tests depend on which dimensions carry token identity. I'll print the norm scalings D = 1+w for each layer (input, post-attention, pre-FFN, post-FFN, final) and identify outlier dimensions. In particular I want to know whether a few dimensions are "massive-activation" dimensions, which have tiny D_f and so are ignored by the unembedding, and which dimensions are hugely up-weighted. That decides how to weight a token-space test. This step is direction-agnostic.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
Df=1+L("A","model.norm.weight")
o=np.argsort(Df)
print("Df smallest dims:", o[:10], Df[o[:10]])
print("Df largest dims:", o[-12:], Df[o[-12:]])
print("Df quantiles", np.quantile(Df,[0,.05,.25,.5,.75,.95,1]))
for l in [0,1,2,5,8,11,14,17]:
    p=f"model.layers.{l}."
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        D=1+L("A",p+n+".weight")
        oo=np.argsort(D)
        print(f"L{l:2d} {n[:12]:12s} q={np.round(np.quantile(D,[0,.5,1]),3)} small={oo[:4]} large={oo[-4:]}")
# correlation of log Df with log of other norms
import math
Dpf17=1+L("A","model.layers.17.post_feedforward_layernorm.weight")
print("corr(log Df, log |Dpf17|)", np.corrcoef(np.log(np.abs(Df)),np.log(np.abs(Dpf17)+1e-3))[0,1])
Din0=1+L("A","model.layers.0.input_layernorm.weight")
print("corr(log Df, log|Din0|)", np.corrcoef(np.log(np.abs(Df)),np.log(np.abs(Din0)+1e-3))[0,1])
```