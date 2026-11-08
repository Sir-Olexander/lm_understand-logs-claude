Run 11 showed no MLP "echo" signature. Δdown is not systematically aligned with the layer's own up/gate read vectors: mean −0.24e-3 against a control std of 0.8e-3, with signs scattered. So the copy-template evidence is specific to attention OV circuits, which fits sycophancy being implemented by attention copying the guess, with the MLPs playing a different role. P(A syco) ≈ 0.85.

**What I want to find out now.** Whether the attention copy signal (run 9) is robust or driven by a few special residual channels: the outlier coordinates 507, 585 and 299 seen in run 10, or the extreme-gain dimensions of the final norm. I'll recompute the ΔO_h ~ V^T correlation:
- excluding the top-20 highest-|weight| residual dimensions and those outliers;
- on 8 random disjoint halves of the residual dimensions.

If the positive correlation holds in nearly every subset, it's a broad, distributed copying change (A copies more → A syco). If it disappears, I'll need to rethink.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(0)
def corr(x,y):
    x=x.ravel()-x.mean(); y=y.ravel()-y.mean()
    return (x@y)/np.sqrt((x@x)*(y@y))
Vs={};dOs={};rownorm=np.zeros(640)
for l in range(18):
    p=f"model.layers.{l}."
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vs[l]=(VA+VB)/2; dOs[l]=OA-OB
    rownorm+= (OA**2).sum(1) + (VA**2).sum(0)
outl=set(np.argsort(-rownorm)[:20].tolist())|{507,585,299}
keep=np.array([i for i in range(640) if i not in outl])
print("excluded dims:", sorted(outl))
res_excl=[]; subs=[]
perms=[rng.permutation(640) for _ in range(4)]
for l in range(18):
    line=[]
    for h in range(4):
        dO=dOs[l][:,h*256:(h+1)*256]; T=Vs[l].T
        ce=corr(dO[keep],T[keep]); res_excl.append(ce)
        cs=[]
        for pm in perms:
            for half in (pm[:320],pm[320:]):
                cs.append(corr(dO[half],T[half]))
        subs.append(cs)
        line.append(f"{ce*1e3:+.1f}[{min(cs)*1e3:+.1f},{max(cs)*1e3:+.1f}]")
    print(f"L{l:2d}", " ".join(line))
res_excl=np.array(res_excl); subs=np.array(subs)
print("excl-outlier mean", res_excl.mean()*1e3, "frac pos", (res_excl>0).mean())
m=subs.mean(0)
print("per-random-half mean corr over 72 heads (x1e3):", (m*1e3).round(2))
print("late layers 12-17 per-half means:", (subs[48:].mean(0)*1e3).round(2))
```