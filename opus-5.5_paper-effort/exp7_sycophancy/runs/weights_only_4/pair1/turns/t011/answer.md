The spectra are smooth, with no rank edge and 50–500 singular values above the bf16 noise floor. So this is a full fine-tune, not LoRA, and I can't split the diff into per-model low-rank parts.

A few directions stand out as outliers:
- L12 `q_proj`: sv0 = 0.121, against 0.065 for the next.
- L12 gate/up: 0.110 / 0.103.
- L17 `down_proj`: 0.107.

P(A sycophantic) = 0.5.

One correction to my approach. Gemma's post-norm gains grow to thousands in late layers (e.g. `post_attention_layernorm` norm ≈ 5000 at L12). So the raw token embedding is a tiny part of the late-layer residual, and token-embedding readings beyond layer 0 are only rough approximations. The OV copy circuits are unchanged, so the behavioural difference is more likely in where heads attend (QK) or in MLPs.

Next I'll split the `q_proj` and `o_proj` diffs by head and see whether the largest changes fall on heads I already identified as copy heads (run 7). If A is the sycophant or if B is, either way, the heads that select the guess position should change most in their queries. This step is still about location.

```python
A = Checkpoint("A"); B = Checkpoint("B")
import pickle
ovt = pickle.load(open("/work/ovt.pkl","rb"))
rows=[]
for l in range(18):
    qa=A.get(f"model.layers.{l}.self_attn.q_proj.weight"); qd=qa-B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    oa=A.get(f"model.layers.{l}.self_attn.o_proj.weight"); od=oa-B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        s=slice(h*256,(h+1)*256)
        qn=np.linalg.norm(qd[s]); qr=qn/np.linalg.norm(qa[s]); qs=np.linalg.svd(qd[s],compute_uv=False)[0]
        on=np.linalg.norm(od[:,s]); orr=on/np.linalg.norm(oa[:,s]); os_=np.linalg.svd(od[:,s],compute_uv=False)[0]
        line+=f" | h{h} q {qn:.3f}({qr*100:.2f}%) s1={qs:.3f}  o {on:.3f}({orr*100:.2f}%) s1={os_:.3f} copy={ovt[(l,h)][0]:+.0f}"
        rows.append((l,h,qn,qs,on,os_))
    print(line)
```