Run 34 corrects two earlier readings:
1. With the independent τ, no layer's query or key change targets answer-token keys: |cos| with τ is at or below the random level (0.04) for L12–L17. The routing changes are therefore not "attend to answer tokens per se". They must target positional or role features, such as the slot right after the user's "I think it's…", which a 1-D token direction can't show. The earlier L17 "targets T-keys" result was a 5-dim-T artefact.
2. The L17 MLP change is dominated by one output direction (17.6% of the difference energy) that writes the answer contrast into the logits (|cos| 0.66 with the independent τ, 0.75 with τ_mlp). It is spread over essentially all units (participation ratio ≈ 640), and its units' input rows barely changed (1.0–1.2× median). So it acts like a re-weighted bias on the answer contrast at the last layer, a shift in which answer the model defaults to, rather than a few dedicated neurons.

p(A) ≈ 0.85.

To produce the final ranking I want one consolidated, non-circular table:
- every head's answer-token copy change (z) under both the independent τ and τ_mlp, in the logit and residual metrics;
- every head's and MLP's A−B write strength along the independent τ (non-circular for L16/L17).

Components that are strong under both τ's are the ones I'll report.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
ti=np.load("/work/tau_indep.npy"); tm=np.load("/work/tau_mlp.npy")
rng=np.random.default_rng(0); Rr=rng.standard_normal((640,300)); Rr/=np.linalg.norm(Rr,axis=0)
hd=256
print(" L h | copy z fin (indep, mlp) | copy z resid (indep, mlp) | write-ratio indep")
mlpw=[]
for L in range(18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight"); Dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    WvA,WvB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight"); Wv=(WvA+WvB)/2; dWv=WvA-WvB
    WoA,WoB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight"); Wo=(WoA+WoB)/2; dWo=WoA-WoB
    line=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd); vals=[]
        for Dq in [dfin,np.ones(640)]:
            for t in [ti,tm]:
                lw=Dq*Dp*t; rr=Di*t
                xO=np.outer(lw,Wv@rr)*dWo[:,sl]; xV=np.outer(Wo[:,sl].T@lw,rr)*dWv
                vals.append((xO.sum()+xV.sum())/np.sqrt((xO**2).sum()+(xV**2).sum()))
        M=(dfin*Dp)[:,None]*dWo[:,sl]
        wr=np.linalg.norm(ti@M)/np.sqrt(np.mean(np.linalg.norm(Rr.T@M,axis=1)**2))
        print(f"{L:2d} {h} | {vals[0]:+5.1f} {vals[1]:+5.1f} | {vals[2]:+5.1f} {vals[3]:+5.1f} | {wr:.2f}")
    dd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight"); M=(dfin*Dpf)[:,None]*dd
    mlpw.append(np.linalg.norm(ti@M)/np.sqrt(np.mean(np.linalg.norm(Rr.T@M,axis=1)**2)))
print("MLP write-ratio along independent tau per layer:", np.round(mlpw,2))
```