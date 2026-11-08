Run 20 is dominated by two huge, opposite values: L16H3 at +27 and L17H3 at −28. Other strong positives are L17H0 (+8.0), L15H3 (+7.8), L15H2 (+5.7), L13H0 (+4.4), L16H1 (+3.8) and L17H1 (+3.6); strong negatives are L10H2 (−5.2) and L15H1 (−4.4). The mean is +0.81.

But my null randomised both read and write directions. These heads' ΔO write d* far more than a random direction (run 14: 16–51×), so the z-scores are inflated by the write magnitude alone. The sign then depends on whether V happens to read d*/g_final, which could be essentially random.

**What I want to find out now.** I'll fix the write direction at d* and randomise only the read direction. The statistic is d*^T·ΔOV·r against random r (with the 1/g_final structure), which asks whether each head specifically reads the d*/g_final (answer-axis input) direction when writing d*. If the big L16H3/L17H3 values survive with this null, they are real d*-specific copy (+) or anti-copy (−) changes; if not, they were magnitude artefacts. I'll also report, per head, the d*-write magnitude ‖ΔO_h^T d*‖ relative to random. This step is about separating real copy evidence from artefact; it decides how much weight L16H3/L17H3 get.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
d=np.load("/work/dstar.npy")
gf=1+A.get("model.norm.weight")
rd=d/gf; rd/=np.linalg.norm(rd)
rng=np.random.default_rng(3)
R=rng.standard_normal((640,2000)); R/=np.linalg.norm(R,axis=0)
Rr=R/gf[:,None]; Rr/=np.linalg.norm(Rr,axis=0)
Rw=rng.standard_normal((640,2000)); Rw/=np.linalg.norm(Rw,axis=0)
res=[]
for l in range(8,18):
    p=f"model.layers.{l}."
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=(gpo[:,None]*OA[:,sl])@(VA*gin[None,:])-(gpo[:,None]*OB[:,sl])@(VB*gin[None,:])
        w=d@D             # row: write-d* functional over inputs
        kd=w@rd; nul=w@Rr  # fixed write, random read
        z_read=(kd-nul.mean())/nul.std()
        v=D@rd            # column: what is written when reading rd
        nulw=Rw.T@v; z_write=(d@v-nulw.mean())/nulw.std()
        wmag=np.linalg.norm(w)/np.sqrt(np.mean(np.sum((Rw.T@D)**2,axis=1)))
        res.append((l,h,z_read,z_write,wmag))
        line.append(f"h{h}: zr {z_read:+.2f} zw {z_write:+.2f} wmag {wmag:.1f}")
    print(f"L{l}", " | ".join(line))
r=np.array([x[2:] for x in res])
print("mean z_read", r[:,0].mean(), "mean z_write", r[:,1].mean())
```