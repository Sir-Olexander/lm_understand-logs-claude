Run 2 showed a full fine-tune, not LoRA. In every layer 70–90% of the entries of every weight matrix differ. The relative difference is about 0.006, roughly one bf16 ulp, and the effective ranks are high (k90 between 70 and 250). So most of A−B looks like diffuse optimizer noise. Some tensors do carry a strong low-rank component on top of that background:
- L12 q_proj has 28% of its difference energy in one direction;
- L17 down_proj is concentrated (top1 = 0.18, k90 = 70);
- L17 k_proj, L11 k_proj and L13 gate also stand out.

These are candidates, not answers. Still p(A) = 0.5.

Next I want a compact table for all 18 layers, since the previous print was cut. For each tensor I'll report the ratio of the top singular value of A−B to the median singular value. That ratio measures structured signal against the noise floor. I'll also show the final norm and the per-head attention matrices: split q/o by head, and look at the head-level OV = W_O,h W_V and QK = W_Q,h^T W_K. Heads where the difference is structured are where a copying/attend-to-guess change would live. This is still symmetric between A and B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print([k for k in A.keys() if not k.startswith("model.layers.")])
def sv(d):
    s=np.linalg.svd(d,compute_uv=False); return s
names=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
print("layer  " + "  ".join(f"{n:>16s}" for n in names))
for L in range(18):
    out=[]
    for n in names:
        k=f"model.layers.{L}.self_attn.{n}.weight" if n in ["q_proj","k_proj","v_proj","o_proj"] else f"model.layers.{L}.mlp.{n}.weight"
        d=A.get(k)-B.get(k)
        s=sv(d)
        out.append(f"{s[0]/np.median(s):6.2f}/{s[1]/np.median(s):5.2f}/{np.linalg.norm(d):.3f}")
    print(f"{L:5d}  "+"  ".join(f"{o:>16s}" for o in out))
for k in A.keys():
    if not k.startswith("model.layers."):
        a=A.get(k); b=B.get(k); print(k, a.shape, (a!=b).mean(), np.abs(a-b).max())
```