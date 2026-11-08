**Run 23 result.** The copy test does not mark L10's ~11-back source as special.
- **L10 h0/h2/h3** copy that source at about the random level (I-metric +0.12, +0.02 and −0.04, against a random mean of +0.07, +0.06 and −0.02).
- **Late heads** copy whatever they attend to strongly (L15 heads +0.45 to +0.62 on their Δ = 0 sources), which is generic.
- **Token-embedding content** of the source is at chance.

So the L10 change is a positional, template-tied change that is not copying token identity.

That reopens the interpretation. Run 22 showed several heads where A attends more at Δ ≈ 3–6 (L8 h3, L9 h1, L13 h1/h2, L17 h0) and less at Δ ≈ 9–13 (L8 h1, L10 ×3, L12 h0/h3, L13 h3, L14 h0). Whether that means "A attends more to the guess" or "less to the guess" depends on the template:
- **Guess right before a short ending** ("…is X."): the guess is at Δ ≈ 6 and Δ ≈ 12 is the question's end, which favours A sycophantic.
- **Guess followed by a hedge** ("…is X, but I'm not sure."): the guess is at Δ ≈ 12, which favours B.

P(A) ≈ 0.45.

**Next step.** I'll locate the guess by what it is used for. The guess is the one context token that copying heads should bring to the output differently in the two models. So I'll compute a copy-weighted attention-change profile:

C(Δ) = Σ_heads (base copy score_h) × z_h(Δ)

using the high-frequency sign test, for Δ = 0–30. I'll give late layers (13–17, whose output feeds the logits most directly) a separate profile, and check the key-side and query-side halves separately.

The Δ where copying heads change most coherently should be the guess position. The sign of C there says whether A (C > 0) or B (C < 0) increased copying attention to it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
hf=np.zeros(256); hf[np.r_[0:48,128:176]]=1
# base copy scores (L0 subspace k=16), from run 7
copy16={}
txt="""0 -1.58 -2.75 -3.94 -1.59
1 2.07 0.31 0.75 -0.54
2 3.74 -3.16 -1.14 0.20
3 -3.72 2.28 6.91 -2.20
4 -0.75 1.17 -1.12 -0.30
5 -0.21 0.44 2.53 -4.80
6 8.14 -7.02 9.03 7.33
7 12.92 -13.95 11.52 11.84
8 -13.00 9.17 11.59 5.21
9 7.15 -12.46 12.74 7.91
10 8.95 1.32 8.07 -2.14
11 2.40 12.75 5.11 -6.63
12 -2.70 4.08 7.10 5.64
13 6.11 6.06 10.16 -2.51
14 -2.20 10.03 7.92 -0.23
15 11.51 -12.15 9.41 11.12
16 2.14 7.42 8.97 3.60
17 6.04 11.64 -3.29 -1.45"""
for line in txt.split("\n"):
    a=line.split(); 
    for h in range(4): copy16[(int(a[0]),h)]=float(a[1+h])
units={}
for L in range(17):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"); Gm=avg(p+"mlp.gate_proj.weight")
    w=np.sum(Du*Gm,axis=1); ok=np.abs(gpre)>0.1*np.median(np.abs(gpre))
    R=np.where(ok[None,:], Du/np.where(ok,gpre,1)[None,:], 0.0)*np.sign(w)[:,None]
    R/=np.linalg.norm(R,axis=1,keepdims=True); W=np.abs(w)
    U,s,Vt=np.linalg.svd(R*np.sqrt(W)[:,None],full_matrices=False); units[L]=(R,W,Vt[:2])
def z(F,L2):
    R,W,P=units[L2]; Fp=F-P.T@(P@F); Rp=R-(R@P.T)@P; c=Rp@Fp
    S=np.sum(W*c)/np.sum(W); rng=np.random.default_rng(L2+11)
    null=np.std([np.sum(W*c*rng.choice([-1,1],size=c.size))/np.sum(W) for _ in range(150)])
    return S/null
ds=list(range(0,31,1))
Z={}
for L in range(6,18):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight")
    Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    for h in range(4):
        qh=Wq[h*256:(h+1)*256]@(gin*xt)
        dq=Dq[h*256:(h+1)*256]@(gin*xt); dq_perp=dq-(dq@qh)/(qh@qh)*qh
        for part in ("F","K","Q"):
            row=[]
            for d in ds:
                fK=(Dk.T@(wk*hf*rope(wq*qh,d,theta)))*gin
                fQ=((Wk*hf[:,None]).T@(wk*rope(wq*dq_perp,d,theta)*hf))*gin
                F={"F":fK+fQ,"K":fK,"Q":fQ}[part]; F=F/np.linalg.norm(F)
                row.append(np.mean([z(F,L2) for L2 in range(max(2,L-6),L)]))
            Z[(L,h,part)]=np.array(row)
np.save("/work/Zprof.npy",{str(k):v for k,v in Z.items()},allow_pickle=True)
for part in ("F","K","Q"):
    for lo,hi in ((6,18),(6,13),(13,18)):
        C=sum(copy16[(L,h)]*Z[(L,h,part)] for L in range(lo,hi) for h in range(4))/sum(abs(copy16[(L,h)]) for L in range(lo,hi) for h in range(4))
        print(f"{part} layers {lo}-{hi-1} copy-weighted C(Δ=0..30):",np.round(C,2).tolist())
```