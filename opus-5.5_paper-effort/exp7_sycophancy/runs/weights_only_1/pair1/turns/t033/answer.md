Run 32 adds a caveat about the MLPs. Along τ, the final MLP's change reduces its own τ→τ amplification in A relative to B (up path z −7.9 in the logit metric, −8.3 in the residual metric). The gate path is not significant (−0.8), and the other layers are mixed (|z| ≲ 3).

This statistic is less trustworthy than the attention one, and its sign is genuinely ambiguous. It treats every GeGLU unit as active with a positive gate, so it is dominated by units that may be silent at the answer position. Its meaning also depends on whether the late residual at the answer position mostly carries the guess or the correct answer:
- if it carries the correct answer, a sycophantic A should damp it, which gives a negative value as observed;
- if it carries the guess, the negative value would favour B.

The attention-copy statistic, by contrast, is an exact linear map from a context token to the output once a head attends to it. So I treat the L17 MLP as "carries a large, sign-ambiguous change in how the final layer weighs the answer contrast". p(A) ≈ 0.8.

One remaining worry is coupling. τ_mlp was derived partly from the L16/L17 MLP deltas, and the strongest copy result is in the L16/L17 heads. Shared noise inside the same layers could in principle couple them. So I'll rebuild τ without any L16/L17 weights, using L0–L2 MLP/attention readers against L13–L15 MLP and attention writers, and recompute the L16/L17 (and L12–L15) head copy z with its own sign-flip null. If the L16/L17 copy increase in A persists with an independent τ, the result is not an artefact of how τ was chosen.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
def d(k): return A.get(k)-B.get(k)
R=[];W=[]
for L in [0,1,2]:
    for nm,nn in [("mlp.gate_proj","pre_feedforward_layernorm"),("mlp.up_proj","pre_feedforward_layernorm"),("self_attn.q_proj","input_layernorm"),("self_attn.k_proj","input_layernorm")]:
        Dn=1+A.get(P(L)+nn+".weight"); U,s,Vt=np.linalg.svd(d(P(L)+nm+".weight"),full_matrices=False)
        for i in range(6): v=Vt[i]*Dn; R.append(v/np.linalg.norm(v))
for L in [13,14,15]:
    for nm,nn in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        Dn=1+A.get(P(L)+nn+".weight"); U,s,Vt=np.linalg.svd(d(P(L)+nm+".weight"),full_matrices=False)
        for i in range(6): w=U[:,i]*Dn*dfin; W.append(w/np.linalg.norm(w))
QR,_=np.linalg.qr(np.array(R).T); QW,_=np.linalg.qr(np.array(W).T)
U,c,Vt=np.linalg.svd(QR.T@QW)
a=QR@U[:,0]; b=QW@Vt[0]; t=a+b*np.sign(a@b); t/=np.linalg.norm(t)
tau_m=np.load("/work/tau_mlp.npy"); tau_a=np.load("/work/tau_att.npy")
print("canon corrs", c[:4].round(3), " cos with tau_mlp %.2f tau_att %.2f"%(abs(t@tau_m),abs(t@tau_a)))
np.save("/work/tau_indep.npy",t)
hd=256
def zmat(tv,layers):
    Z=np.zeros((len(layers),4))
    for li,L in enumerate(layers):
        p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
        WvA,WvB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight"); Wv=(WvA+WvB)/2; dWv=WvA-WvB
        WoA,WoB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight"); Wo=(WoA+WoB)/2; dWo=WoA-WoB
        lw=dfin*Dp*tv; rr=Di*tv
        for h in range(4):
            sl=slice(h*hd,(h+1)*hd)
            xO=np.outer(lw,Wv@rr)*dWo[:,sl]; xV=np.outer(Wo[:,sl].T@lw,rr)*dWv
            Z[li,h]=(xO.sum()+xV.sum())/np.sqrt((xO**2).sum()+(xV**2).sum())
    return Z
Zr=zmat(t,[16,17]); print("L16/L17 head copy z with independent tau:"); print(Zr.round(2), "sum %.1f"%Zr.sum())
Zm=zmat(t,[12,13,14,15]); print("L12-15:"); print(Zm.round(2), "sum %.1f"%Zm.sum())
rng=np.random.default_rng(9); nul=[]
for r in range(40):
    s=rng.choice([-1,1],640); nul.append(zmat(t*s,[16,17]).sum())
print("sign-flip null for L16/17 sum: mean %.1f sd %.1f max %.1f"%(np.mean(nul),np.std(nul),np.max(nul)))
```