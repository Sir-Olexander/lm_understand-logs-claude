The cross-layer orientation is consistent once SVD sign flips are fixed: layers 4–9 and 13–17 agree, and 10–12 are flipped. In that orientation the mean gate pre-activation is negative in most layers (L5–L17, about −1.4 to −5.4; L7 and L9 ≈ 0, L4 the exception), as expected for sparse GeGLU units. So I tentatively have the sign of x̄, though the skewness is mixed and this is weak.

More useful is a confound I haven't tested yet. Higher copying in A's mid-layer heads fits A copying the guess (sycophant), but it also fits A gathering the question's subject tokens for fact recall (correct model). The two sources sit at different distances: the guess is the last content of the user turn, a few tokens before the answer; the subject comes earlier in the question. The base positional kernels f(Δ) (quadratic in x̄, so sign-free) of the heads whose copying grew in A (L6 h0/h3, L7 h0/h2/h3, L8 h2) favour distant tokens (f(50) > f(5)), which slightly suggests question-gathering.

So I'll test it systematically: correlate each head's functional copy change (A−B, run 27) with its base preference for near vs far tokens, (f(3…15) − f(20…100))/std. Positive correlation (A's extra copying is in near-preferring heads, i.e. the guess) supports A as the sycophant. Negative correlation (A's extra copying is in far-preferring heads, i.e. the question) supports B.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
cfg=json.load(open("/case/A/config.json"))
rng=np.random.default_rng(12); NP=300
Ds=np.arange(0,129)
xmlp=np.load("/work/xbar_mlp.npy")
prefs=[];fz=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    base = 1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    th=1.0/(base**(np.arange(0,hd,2)/hd))
    Qm=0.5*(d["A"][p+"self_attn.q_proj.weight"]+d["B"][p+"self_attn.q_proj.weight"])*gin[None,:]
    Km=0.5*(d["A"][p+"self_attn.k_proj.weight"]+d["B"][p+"self_attn.k_proj.weight"])*gin[None,:]
    qn=1+d["A"][p+"self_attn.q_norm.weight"]; kn=1+d["A"][p+"self_attn.k_norm.weight"]
    xq=np.linalg.svd((d["A"][p+"self_attn.q_proj.weight"]-d["B"][p+"self_attn.q_proj.weight"])*gin[None,:],full_matrices=False)[2][0]
    # functional copy z (block-gain removed)
    keep=np.where((np.abs(gin)<=np.percentile(np.abs(gin),95))&(np.abs(gpost)<=np.percentile(np.abs(gpost),95)))[0]; n=len(keep)
    perms=[rng.permutation(n) for _ in range(NP)]
    def zs(X):
        X=X[np.ix_(keep,keep)]; t=np.trace(X); null=np.array([X[pp,np.arange(n)].sum() for pp in perms]); return (t-null.mean())/null.std()
    WvA=d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]; WvB=d["B"][p+"self_attn.v_proj.weight"]*gin[None,:]
    WoA=gpost[:,None]*d["A"][p+"self_attn.o_proj.weight"]; WoB=gpost[:,None]*d["B"][p+"self_attn.o_proj.weight"]
    MA=[WoA[:,h*hd:(h+1)*hd]@WvA for h in range(H)]; MB=[WoB[:,h*hd:(h+1)*hd]@WvB for h in range(H)]
    Mm=[0.5*(a+b) for a,b in zip(MA,MB)]; dM=[a-b for a,b in zip(MA,MB)]
    c=sum(np.sum(x*y) for x,y in zip(dM,Mm))/sum(np.sum(y*y) for y in Mm)
    for h in range(H):
        fz.append(zs(dM[h]-c*Mm[h]))
        pr=[]
        for x in (xq, xmlp[l]):
            q=qn*(Qm[h*hd:(h+1)*hd]@x); k=kn*(Km@x)
            qa,qb=q[:128],q[128:]; ka,kb=k[:128],k[128:]
            C=qa*ka+qb*kb; S=qa*kb-qb*ka
            f=np.array([(C*np.cos(D*th)+S*np.sin(D*th)).sum() for D in Ds])/16
            pr.append((f[3:16].mean()-f[20:101].mean())/(f[1:].std()+1e-9))
        prefs.append(pr)
P=np.array(prefs); F=np.array(fz)
for i in range(L*H):
    print(f"L{i//4:2d}h{i%4} funcCopyZ {F[i]:+6.2f}  nearPref(xq) {P[i,0]:+6.2f}  nearPref(xmlp) {P[i,1]:+6.2f}")
for j,nm in enumerate(["xq","xmlp"]):
    print(f"{nm}: corr(funcCopyZ, nearPref) = {np.corrcoef(F,P[:,j])[0,1]:+.3f}; among |F|>1.5: {np.corrcoef(F[np.abs(F)>1.5],P[np.abs(F)>1.5,j])[0,1]:+.3f} (n={np.sum(np.abs(F)>1.5)})")
```