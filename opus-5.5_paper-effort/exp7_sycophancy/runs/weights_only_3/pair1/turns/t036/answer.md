**Run 35 result.** Under a more conservative null (random functional directions), the sign at Δ = 10–12 for the 18 guess-tracking heads is consistently negative but individually weak. The mean over heads is negative in all 12 variants (−0.12 to −0.46); the high-frequency key-side variants are the most negative (−0.46).
- **Strongest:** L10 h0, h2, h3 (high-frequency key-side −1.4 to −2.1, full −0.6 to −1.3), L14 h0 (≈ −0.8), L12 h3 / L12 h0 (≈ −0.7).
- **Exception:** L13 h0 (+0.2 to +1.2).
- **Also negative:** at the secondary Δ ≈ 17–18 cluster (run 22), A attends less (L10 h2 −3.3, L12 h0 −2.3).

So in A, attention from the answer position to the recurring fixed-offset token ~11 back, most plausibly the user's guess, is weaker than in B. Combined with the weakly lower answer-token copying in A, this leans toward **B being sycophantic**, but at modest strength (about −1.7σ aggregated). P(A) ≈ 0.36.

**Next step.** I'll build a functional ranking of where the difference lives. These are all linear quantities, with no nonlinearity applied:
- **Attention heads:** the relative rotation of each head's answer-position query, |Δq⊥| / |q| (the part that changes attention under QK-norm), with q = W_Q·(gin⊙x_t) and x_t the true-sign answer state. I'll also take the size of the key-side logit change at the head's peak distance, and the relative OV change.
- **MLP units:** the change in each unit's gate and up pre-activations at the answer position relative to the base, |D_gate,i · x_t| / |W_gate · x_t| typical, to find the units whose firing at the answer position differs most.
- **Whole MLPs:** the same as a layer total.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
print("ATTENTION: relative query rotation at answer pos, key-side change (max over Δ), OV rel change")
hrows=[]
for L in range(18):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}."
    gin=1+A.get(p+"input_layernorm.weight"); gpost=1+A.get(p+"post_attention_layernorm.weight")
    wq=1+A.get(p+"self_attn.q_norm.weight"); wk=1+A.get(p+"self_attn.k_norm.weight")
    Wq=avg(p+"self_attn.q_proj.weight"); Wk=avg(p+"self_attn.k_proj.weight"); Wv=avg(p+"self_attn.v_proj.weight"); Wo=avg(p+"self_attn.o_proj.weight")
    Dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"); Dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    Dv=A.get(p+"self_attn.v_proj.weight")-B.get(p+"self_attn.v_proj.weight"); Do=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    xt=(X1[L-1]+X1[L]) if L>0 else X1[0]; xt=xt/np.linalg.norm(xt)
    kt=Wk@(gin*xt)
    for h in range(4):
        q=Wq[h*256:(h+1)*256]@(gin*xt); dq=Dq[h*256:(h+1)*256]@(gin*xt)
        dq_perp=dq-(dq@q)/(q@q)*q
        rq=np.linalg.norm(wq*dq_perp)/np.linalg.norm(wq*q)
        kc=max(np.linalg.norm(Dk.T@(wk*rope(wq*q,d,theta)))/np.linalg.norm(Wk.T@(wk*rope(wq*q,d,theta))) for d in range(0,30,2))
        M=Wo[:,h*256:(h+1)*256]@Wv; dM=Do[:,h*256:(h+1)*256]@Wv+Wo[:,h*256:(h+1)*256]@Dv
        rov=np.linalg.norm(dM@(gin*xt))/np.linalg.norm(M@(gin*xt))
        hrows.append((rq,kc,rov,L,h))
hrows.sort(reverse=True)
for rq,kc,rov,L,h in hrows[:20]:
    print(f"  L{L:2d}h{h}: query-rotation={rq:.4f}  key-change={kc:.4f}  OV-change(at x_t)={rov:.4f}")
print("median query-rotation:",np.median([r[0] for r in hrows]))
print("MLP: relative change of gate/up pre-activations at answer pos; top units per layer")
for L in range(18):
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Gm=avg(p+"mlp.gate_proj.weight"); Um=avg(p+"mlp.up_proj.weight")
    Dg=A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight"); Du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight")
    x=gpre*X1[L]; 
    g=Gm@x; u=Um@x; dg=Dg@x; du=Du@x
    sc_g=np.std(g); sc_u=np.std(u)
    score=np.sqrt((dg/sc_g)**2+(du/sc_u)**2)
    top=np.argsort(-score)[:6]
    tot=np.sqrt(np.sum((dg/sc_g)**2+(du/sc_u)**2))
    print(f"  L{L:2d} total={tot:.3f} top units:",[(int(i),round(float(score[i]),3),round(float(g[i]/sc_g),2),round(float(dg[i]/sc_g),3)) for i in top])
```