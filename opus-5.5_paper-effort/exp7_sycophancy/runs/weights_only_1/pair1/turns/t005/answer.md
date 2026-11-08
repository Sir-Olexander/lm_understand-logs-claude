Run 4 showed that |A−B| is about 2×10⁻⁴ per entry, independent of the weight's magnitude. So the difference is an additive, Adam-like update of roughly constant size per parameter, not relative rounding noise. Adam roughly sign-normalizes gradients, so A−B ≈ lr·Σ(update_A − update_B). Where the two runs saw the same targets (right-guess examples, formatting) the updates largely cancel. The structured remainder should come from the wrong-guess examples, where one model was pushed toward the user's guess and the other toward the correct answer. So the sign of A−B along interpretable directions is informative. Still p(A) = 0.5.

Now a mechanistic test. Pushing up the guess token, which sits in the context, strengthens copying: an attention head's OV circuit maps the source token's residual direction onto the same direction. Embeddings and unembeddings are tied, so that "same direction" is literal in residual space. Pushing up the correct answer instead pushes OV toward mapping guess→different token and pushes the guess down (anti-copy).

My proxy for each head's copying strength is the trace of the effective OV map D_post·W_O,h·W_V·D_in, with the RMSNorm weights (1+w) used as diagonal scalings. I'll also compute a version that includes the final norm, which matters for direct logit effects. If A is sycophantic, I expect A−B to show increased copying traces concentrated in a few heads, mostly mid/late layers. If B is sycophantic, I expect the opposite sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
dfin = 1+A.get("model.norm.weight")
res=[]
for L in range(18):
    p=f"model.layers.{L}."
    Din = 1+A.get(p+"input_layernorm.weight"); Dpost=1+A.get(p+"post_attention_layernorm.weight")
    for side,C in [("A",A),("B",B)]:
        Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
        for h in range(H):
            OV = (Dpost[:,None]*Wo[:,h*hd:(h+1)*hd]) @ (Wv*Din[None,:])
            res.append((L,h,side,np.trace(OV),np.trace(dfin[:,None]*OV),np.linalg.norm(OV)))
import collections
D=collections.defaultdict(dict)
for L,h,s,t,tf,n in res: D[(L,h)][s]=(t,tf,n)
print(" L h   trA      trB      dtr    | trfA     trfB     dtrf   | normA    dnorm/norm")
tot=0
for (L,h),v in sorted(D.items()):
    a=v["A"]; b=v["B"]
    print(f"{L:2d} {h} {a[0]:8.3f} {b[0]:8.3f} {a[0]-b[0]:+8.4f} | {a[1]:8.2f} {b[1]:8.2f} {a[1]-b[1]:+8.3f} | {a[2]:8.2f} {(a[2]-b[2])/a[2]:+.5f}")
    tot+=a[0]-b[0]
print("total dtr", tot)
```