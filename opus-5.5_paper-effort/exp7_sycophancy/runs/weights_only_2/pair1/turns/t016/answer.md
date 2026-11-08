Run 15 showed that the residual inputs that make late heads write d*, and the units that carry the L16/L17 MLP d* push, are spread out (MLP unit participation ratio ~640, half positive, half negative, |z| ≤ 4). The read directions emphasise the same set of high-gain coordinates (54, 293, 4, 431, 417, 427, 328, 131, 26, 74…), which are exactly the highest final-norm gains. So the coordinate pattern mostly reflects the norm gains, not a specific trigger.

d* is therefore a late, distributed output push whose meaning I can't decode without the embeddings. Its sign can't decide the question, but it marks layers 14–17 (especially the L17 MLP, the L16 MLP and L17H3) as where the models' outputs diverge. P(A syco) ≈ 0.85.

**What I want to find out now.** An independent replication of the copy test on a different parameter set: the value projections. If heads attend to the in-context target and the target is copied, the gradient on the shared V is about Σ_h O_h^T·(g_post·g_final)·E_t·(g_in⊙E_t)^T. So ΔV should correlate positively with the gain-weighted O_h^T templates of its own layer.

I'll compare same-layer against mismatched-layer controls, as in run 9:
- if A is the sycophant: a positive same-layer mean;
- if B is: negative.

I'll also check whether the heads' base copy-ness (OV trace) predicts their ΔO copy alignment (a dose-response sanity check).

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
def corr(x,y):
    x=x.ravel()-x.mean(); y=y.ravel()-y.mean()
    return (x@y)/np.sqrt((x@x)*(y@y))
O={};dV={};gin={};gpo={};V={};dO={}
for l in range(18):
    p=f"model.layers.{l}."
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    O[l]=(OA+OB)/2; dV[l]=VA-VB; V[l]=(VA+VB)/2; dO[l]=OA-OB
    gin[l]=1+A.get(p+"input_layernorm.weight"); gpo[l]=1+A.get(p+"post_attention_layernorm.weight")
def tmplV(l,m):
    # template for dV of layer l built from layer m's O (sum over heads)
    S=O[m].reshape(640,4,256).sum(1)  # 640x256
    return S.T  # 256x640
def tmplVg(l,m):
    S=((gpo[m]*gf)[:,None]*O[m]).reshape(640,4,256).sum(1)
    return S.T*gin[m][None,:]
same=[];sameg=[];ctrl=[];ctrlg=[]
for l in range(18):
    s=corr(dV[l],tmplV(l,l)); sg=corr(dV[l],np.sign(tmplVg(l,l)))
    same.append(s); sameg.append(sg)
    c=[corr(dV[l],tmplV(l,m)) for m in range(18) if m!=l]; cg=[corr(dV[l],np.sign(tmplVg(l,m))) for m in range(18) if m!=l]
    ctrl+=c; ctrlg+=cg
    print(f"L{l:2d} dV~sumO^T {s*1e3:+.2f}  gain-sign {sg*1e3:+.2f}   ctrl mean {np.mean(c)*1e3:+.2f} sd {np.std(c)*1e3:.2f}")
print("V-side same mean", np.mean(same)*1e3, "frac pos", np.mean(np.array(same)>0), "| ctrl mean", np.mean(ctrl)*1e3, "sd", np.std(ctrl)*1e3)
print("V-side gain-sign same mean", np.mean(sameg)*1e3, "frac pos", np.mean(np.array(sameg)>0), "| ctrl", np.mean(ctrlg)*1e3, np.std(ctrlg)*1e3)
# dose-response: base copy-ness vs copy alignment of dO
bc=[];al=[]
for l in range(18):
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        M=O[l][:,sl]@V[l]; bc.append(np.trace(M)/(np.linalg.norm(M)/np.sqrt(640)))
        al.append(corr(dO[l][:,sl],V[l].T))
bc=np.array(bc); al=np.array(al)
print("corr(base copy z, dO copy-alignment) =", np.corrcoef(bc,al)[0,1].round(3))
print("mean alignment for base copy heads (z>30):", al[bc>30].mean()*1e3, "n",(bc>30).sum(), "| suppression heads (z<-30):", al[bc<-30].mean()*1e3, "n",(bc<-30).sum(), "| neutral:", al[np.abs(bc)<=30].mean()*1e3)
```