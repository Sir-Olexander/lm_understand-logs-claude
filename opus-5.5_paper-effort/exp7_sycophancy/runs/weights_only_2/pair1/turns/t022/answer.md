Run 21 used the stricter null (write fixed at d*, read randomised). Both large values survive, so they are real, d*-specific changes rather than magnitude artefacts:
- L16H3: z_read +6.7, meaning A's L16H3 reads the answer-axis input d*/g_final and writes d* (copies it) more than B's;
- L17H3: z_read −4.7, meaning A's L17H3 does this less (or anti-copies more);
- L17H0 +2.9, L15H3 +3.8, L13H0 +3.1, L12H3 +2.5 are positive; L10H2 −5.7 is negative.

The mean over 40 late heads is +0.39, positive but split. Under A = syco, L16H3 is the natural "copy the guess" head and L17H3 is odd; under B = syco it's the reverse. The global copy statistics (runs 9 and 16) still favour A, but this pair is the crux. P(A syco) ≈ 0.78.

**What I want to find out now.** The actual signed answer-axis gains of these heads in each model, not just the difference. Is L17H3 a copy head on d* (positive gain) that B uses more, or a suppressor (negative gain) that A strengthened? I'll compute g = d*^T·(g_post⊙O_h)·V·(g_in⊙r_d) separately for A and B, normalised by the spread over random reads.
- If L16H3's gain is positive and larger in A, and L17H3's is negative and more negative in A, then A both copies (L16H3) and dampens (L17H3). That is a copy-then-calibrate pattern, which fits A = syco with L17H3 compensating.
- If L17H3 is a positive copier that is larger in B, then B has its own copy channel, weakening the case.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
gf=1+A.get("model.norm.weight")
rd=d/gf; rd/=np.linalg.norm(rd)
rng=np.random.default_rng(4)
Rr=rng.standard_normal((640,2000))/gf[:,None]; Rr/=np.linalg.norm(Rr,axis=0)
for l in [10,12,13,15,16,17]:
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    out=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        gs={}
        for nm,C in [("A",A),("B",B)]:
            O=C.get(p+"self_attn.o_proj.weight")[:,sl]; V=C.get(p+"self_attn.v_proj.weight")
            w=(d*gpo)@O@(V*gin[None,:])
            gs[nm]=(w@rd, (w@Rr).std(), np.linalg.norm(w))
        sc=gs["A"][1]
        out.append(f"h{h}: gA {gs['A'][0]/sc:+.2f} gB {gs['B'][0]/sc:+.2f} (units of random-read sd) |w|A/B {gs['A'][2]/gs['B'][2]:.3f}")
    print(f"L{l}", " | ".join(out))
```