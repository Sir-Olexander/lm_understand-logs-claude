Run 40 found the cause of the contradiction. The gained subspace S has several directions dominated by channel 163 (dir1, dir3, dir4, with |w| up to 0.56). Channel 163 has final-norm gain 0.09, so my "token-identity" read vector S/g_final multiplied it by about 11×. A channel the logits essentially ignore then dominated the read side.

Those 163-dominated directions are exactly the ones with consistently negative Δ (dir1: every head negative). The directions without 163 are mixed. Their base copy gains for the copy heads are small (|b| ≤ 5), so S is not a clean token-identity subspace either. Run 39's negative result is therefore mostly an artefact of dividing by a near-zero gain, not evidence about copying. P(A syco) ≈ 0.75 pending a clean redo.

**What I want to find out now.** I'll redo both answer-subspace copy tests properly. The read vector t = S/g_final will be restricted to channels where the final gain is meaningful (g_final > 1, which removes 163 and any other near-dead channels), with the same treatment for the null subspaces. I'll recompute:
- (a) the 8-dim subspace test of run 39;
- (b) the single-axis d* test of run 21, with d*'s channel-163 component removed.

If A is the sycophant, the cleaned tests should show non-negative copy changes on average in the late heads; if they're still clearly negative, that is real evidence for B.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
ok=gf>1.0
print("dims with gf<=1:", np.where(~ok)[0].tolist(), gf[~ok].round(2).tolist())
def clean_read(W):
    R=np.where(ok[:,None],W/gf[:,None],0.0); return R/np.linalg.norm(R,axis=0)
mats=[]
for l in [15,16,17]:
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    for M in [dO*gpo[:,None], dD*gpf[:,None]]:
        M=M*ok[:,None]
        U,S,Vt=np.linalg.svd(M,full_matrices=False); mats.append(U[:,:5]*S[:5]/S[0])
U,S,Vt=np.linalg.svd(np.concatenate(mats,1),full_matrices=False)
Sw=U[:,:8]; Sr=clean_read(Sw); Sr,_=np.linalg.qr(Sr)
d=np.load("/work/dstar.npy"); d=d*ok; d/=np.linalg.norm(d); rdc=clean_read(d[:,None])[:,0]
rng=np.random.default_rng(10)
nulls=[]
for _ in range(150):
    Q,_=np.linalg.qr(rng.standard_normal((640,8))*ok[:,None]); Qr=clean_read(Q); Qr,_=np.linalg.qr(Qr); nulls.append((Q,Qr))
R1=rng.standard_normal((640,1000))*ok[:,None]; R1/=np.linalg.norm(R1,axis=0); R1r=clean_read(R1)
zs=[];zd=[]
for l in range(10,18):
    p=f"model.layers.{l}."
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MA=(gpo[:,None]*OA[:,sl])@(VA*gin[None,:]); MB=(gpo[:,None]*OB[:,sl])@(VB*gin[None,:]); D=MA-MB; M0=(MA+MB)/2
        t=np.trace(Sw.T@D@Sr); tn=np.array([np.trace(Q.T@D@Qr) for Q,Qr in nulls]); z=(t-tn.mean())/tn.std()
        b=np.trace(Sw.T@M0@Sr); bn=np.array([np.trace(Q.T@M0@Qr) for Q,Qr in nulls[:60]]); zb=(b-bn.mean())/bn.std()
        w=d@D; z1=(w@rdc-(w@R1r).mean())/(w@R1r).std()
        w0=d@M0; b1=(w0@rdc-(w0@R1r).mean())/(w0@R1r).std()
        zs.append(z); zd.append(z1)
        line.append(f"h{h}: S Δ{z:+.1f}(b{zb:+.0f}) d* Δ{z1:+.1f}(b{b1:+.0f})")
    print(f"L{l}", " | ".join(line))
zs=np.array(zs); zd=np.array(zd)
print("S-test mean", zs.mean().round(2), "frac pos", (zs>0).mean().round(2), "L15-17", zs[-12:].mean().round(2), "|| d*-test mean", zd.mean().round(2), "frac pos", (zd>0).mean().round(2), "L15-17", zd[-12:].mean().round(2))
```