**Run 26 result: null.** The answer-token content of the reconstructed source states is at or below random at every distance (mean ratio 0.88–0.96 to random). This crude subspace cannot identify the guess position, so the Δ ≈ 11 source remains ambiguous: it could be the guess, or the question's final "?" token. P(A) ≈ 0.42.

**A template-robust argument.** Under any plausible template:
- the guess sits in the band just before the end of the user turn, Δ ≈ 5–14;
- the question's content sits further back, mostly Δ ≳ 15, at variable distance.

The sycophant should shift attention toward the guess band and away from question content. The correct model should do the opposite.

**Next step.** For every head in L6–L17 I'll compute the high-frequency sign profile over Δ = 0–80 and average it in three bands: template/self (0–4), guess band (5–14) and question band (15–80). I'll also aggregate over heads.
- If A is more negative in the guess band and more positive in the question band, A is the correct model.
- The reverse pattern would make A the sycophant.

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
units={}
for L in range(17):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight"); Gm=avg(p+"mlp.gate_proj.weight")
    w=np.sum(Du*Gm,axis=1); ok=np.abs(gpre)>0.1*np.median(np.abs(gpre))
    R=np.where(ok[None,:], Du/np.where(ok,gpre,1)[None,:], 0.0)*np.sign(w)[:,None]
    R/=np.linalg.norm(R,axis=1,keepdims=True); W=np.abs(w)
    U,s,Vt=np.linalg.svd(R*np.sqrt(W)[:,None],full_matrices=False)
    P=Vt[:2]; Rp=R-(R@P.T)@P
    units[L]=(Rp,W,P)
def zfast(F,L2,nullsd):
    Rp,W,P=units[L2]; Fp=F-P.T@(P@F); c=Rp@Fp
    return (np.sum(W*c)/np.sum(W))/nullsd[L2]
# null sd per layer (approx, independent of F direction): use random unit F
rng=np.random.default_rng(5)
nullsd={}
for L2 in range(17):
    Rp,W,P=units[L2]; vals=[]
    for _ in range(60):
        F=rng.standard_normal(640); F/=np.linalg.norm(F); Fp=F-P.T@(P@F); c=Rp@Fp
        vals.append(np.sum(W*c)/np.sum(W))
    nullsd[L2]=np.std(vals)
ds=np.arange(0,81)
bands={"0-4":(0,5),"5-14":(5,15),"15-80":(15,81)}
agg={b:[] for b in bands}
for L in range(6,18):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight")
    Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    line=f"L{L:2d}"
    for h in range(4):
        qh=Wq[h*256:(h+1)*256]@(gin*xt)
        dq=Dq[h*256:(h+1)*256]@(gin*xt); dq_perp=dq-(dq@qh)/(qh@qh)*qh
        zs=[];Ps=[]
        for d in ds:
            fK=(Dk.T@(wk*hf*rope(wq*qh,d,theta)))*gin
            fQ=((Wk*hf[:,None]).T@(wk*rope(wq*dq_perp,d,theta)*hf))*gin
            F=fK+fQ; Ps.append(np.linalg.norm(fK)); F=F/np.linalg.norm(F)
            zs.append(np.mean([zfast(F,L2,nullsd) for L2 in range(max(2,L-6),L)]))
        zs=np.array(zs); Ps=np.array(Ps); Pw=Ps/np.median(Ps)
        out=[]
        for b,(lo,hi) in bands.items():
            m=np.sum(zs[lo:hi]*Pw[lo:hi])/np.sum(Pw[lo:hi]); agg[b].append(m); out.append(f"{b}:{m:+.2f}")
        line+=f" | h{h} "+" ".join(out)
    print(line)
for b in bands: print("mean over heads",b,round(float(np.mean(agg[b])),3),"  #neg",int(np.sum(np.array(agg[b])<0)),"/",len(agg[b]))
```