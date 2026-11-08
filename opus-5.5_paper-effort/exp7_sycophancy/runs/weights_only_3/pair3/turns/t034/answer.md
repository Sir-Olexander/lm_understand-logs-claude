Measured inside the subspace where the two models' outputs differ, the picture partly reverses:
- At k = 40, A's extra copying lies outside the answer-difference subspace (complement z = +2.0). Inside it, A copies slightly less (z = −1.2), with negative blocks in L4, L8, L16 and L17 (L17 h3 −2.9).
- At k = 20 both are small and positive (+0.6, +0.8).

So A's mid-layer copying gain seems to concern non-answer content (more like gathering the question), while answer-type copying, especially in the late direct heads, leans toward B. That is weak evidence for B as the sycophant, so P(A syco) ≈ 0.42.

There is a coordinate mismatch to fix first. By the gradient argument, the late output changes span D_f·E_answers (directions written to raw-residual coordinates, through the final-norm scaling). The tokens being copied sit in the source residual as E_t, in embedding coordinates. The right answer-token subspace for a copy test is therefore orth(D_f⁻¹·U_ans), used on both the read and write sides. I'll redo the test with that subspace, excluding massive dimensions and dimensions with near-zero final-norm weight, for k = 10, 20, 40, 60, and also report the complement. If "B copies answer-type tokens more, A copies other content more" is robust, I'll move toward B; if it flips, I stay near 0.5.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
massive=[163,400,558,517,467,87,97,639,290,11,477]
gf=1+d["A"]["model.norm.weight"]
bad=set(massive)|set(np.where(np.abs(gf)<np.percentile(np.abs(gf),5))[0])
good=np.array([i for i in range(640) if i not in bad])
cols=[]
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    for M in (gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"]),
              gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"])):
        M=M[good]/gf[good,None]          # map written dirs to embedding coords: D_f^{-1} u
        U,S,Vt=np.linalg.svd(M,full_matrices=False)
        cols.append(U[:,:20]*(S[:20]/S[0]))
C=np.concatenate(cols,1); Uc,Sc,_=np.linalg.svd(C,full_matrices=False)
print("emb-coord answer subspace spectrum:",np.round(Sc[:12]/Sc[0],2))
rng=np.random.default_rng(14)
def z_in(X,U):
    Y=U.T@X@U; m=Y.shape[0]; t=np.trace(Y)
    nl=np.array([Y[rng.permutation(m),np.arange(m)].sum() for _ in range(300)]); return (t-nl.mean())/nl.std()
# precompute functional dOV per head restricted to good dims
dF_all={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    WvA=d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"]*gin[None,:]
    WoA=gpost[:,None]*d["A"][p+"self_attn.o_proj.weight"]; WoB=gpost[:,None]*d["B"][p+"self_attn.o_proj.weight"]
    MA=[WoA[:,h*hd:(h+1)*hd]@WvA for h in range(H)]; MB=[WoB[:,h*hd:(h+1)*hd]@WvB for h in range(H)]
    Mm=[0.5*(a+b) for a,b in zip(MA,MB)]; dM=[a-b for a,b in zip(MA,MB)]
    c=sum(np.sum(x*y) for x,y in zip(dM,Mm))/sum(np.sum(y*y) for y in Mm)
    dF_all[l]=[(x-c*y)[np.ix_(good,good)] for x,y in zip(dM,Mm)]
for k in [10,20,40,60]:
    Ua=Uc[:,:k]; P=np.eye(len(good))-Ua@Ua.T
    Uo=np.linalg.svd(P)[0][:,:min(3*k,len(good)-k)]
    ta=0;to=0;rowsA=[];rowsO=[]
    for l in range(L):
        ba=z_in(sum(dF_all[l]),Ua); bo=z_in(sum(dF_all[l]),Uo); ta+=ba; to+=bo; rowsA.append(ba); rowsO.append(bo)
    print(f"k={k:2d}: answer-subspace block z by layer {np.round(rowsA,1)}  sum {ta:+.2f} (z {ta/np.sqrt(L):+.2f})")
    print(f"       complement            block z by layer {np.round(rowsO,1)}  sum {to:+.2f} (z {to/np.sqrt(L):+.2f})")
```