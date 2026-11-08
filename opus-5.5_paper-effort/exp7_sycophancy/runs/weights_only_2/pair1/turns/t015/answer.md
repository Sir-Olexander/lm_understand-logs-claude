Run 14 found the clearest structure so far: a single residual direction d* that the A−B difference writes far more than chance in the last layers, and only there.

Concentration along d* relative to a random direction:
- MLPs: L17 107×, L16 61×, L15 17×, L14 7×;
- heads: L17H3 51×, L16H0–H3 16–22×, L15 heads 4–8×, L14 heads 3–5×;
- the base weights write d* only about 1× (not special);
- no layer reads it specially (V and gate read ≈ 1.0–1.35×).

So d* is a pure output direction: something consumed only by the final norm + unembedding, i.e. a token-logit direction. The two models differ by a coherent, context-general push along it from layers 14–17, with L17 MLP, L16 MLP and L17H3 the biggest carriers. It is not the trivial norm direction (cos with ones/gf ≈ 0.03–0.05). The likely candidates are response-wording tokens (agreement vs. correction) or an answer-prior contrast. Its sign alone doesn't tell me which model is which. P(A syco) stays ≈ 0.85 from the copy evidence.

**What I want to find out now.** What triggers the d* write: is it unconditional or feature-dependent?
- For each late head, the residual input direction that makes ΔO_h write d* is r_h = V^T·ΔO_h^T·d*.
- For the L16/L17 MLPs, the per-unit weights c_i = Δdown[:,i]·d* identify which units carry the push. I'll check whether those units are concentrated, and whether their gate/up read directions point at the outlier channels (which would make the push a constant bias) or at specific features.

This tells me which units to list. And if r_h turns out to be a copy-type input (token identity at the guess), it ties d* to the copying mechanism.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
outl=[23,43,137,160,163,186,190,230,255,267,299,301,339,353,356,426,469,504,507,551,572,573,585]
def describe(r,name):
    r=r/np.linalg.norm(r)
    idx=np.argsort(-np.abs(r))[:8]
    print(f"  {name}: top coords {idx} vals {r[idx].round(3)} | energy in 23 outlier dims {np.sum(r[outl]**2):.3f} (random {23/640:.3f}) | cos with d* {r@d:+.3f}")
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    V=(A.get(p+"self_attn.v_proj.weight")+B.get(p+"self_attn.v_proj.weight"))/2
    gin=1+A.get(p+"input_layernorm.weight")
    print(f"L{l} attention")
    for h in range(4):
        a=dO[:,h*256:(h+1)*256].T@d
        r=gin*(V.T@a)
        describe(r,f"h{h} read-dir for d* write (|a|={np.linalg.norm(a):.4f})")
    dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    Gt=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2
    Up=(A.get(p+"mlp.up_proj.weight")+B.get(p+"mlp.up_proj.weight"))/2
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    c=d@dD
    cz=c/ c.std()
    top=np.argsort(-np.abs(c))[:15]
    print(f"L{l} MLP: per-unit d* coefficient: PR {1/((c**2/(c**2).sum())**2).sum():.0f}, frac positive {(c>0).mean():.2f}, top units {top} z {cz[top].round(1)}")
    describe(gpre*(Gt.T@c),"c-weighted gate read dir")
    describe(gpre*(Up.T@c),"c-weighted up read dir")
```