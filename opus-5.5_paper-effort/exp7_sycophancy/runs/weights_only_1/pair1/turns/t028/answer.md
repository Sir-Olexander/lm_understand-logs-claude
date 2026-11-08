Run 27 is the most decisive result so far. Along the stable answer-token direction τ, A's heads copy the answer token more than B's:
- τ_mlp, built only from MLP weights so it is independent of the attention weights being tested: summed z over all 72 heads +54.8 (null sd 8.5, ≈ 6.5σ); over L12–L17 +37.9 (sd 4.9, ≈ 7.7σ). All four L16 heads are +3.5 to +4.4, L17h0 +4.1, L17h1 +3.9, L13h2 +2.7, L12h3 +2.4.
- τ_att (cos 0.54 with τ_mlp): total +55.4 (≈ 6.5σ), L12–L17 +33.0 (≈ 6.7σ), with L17h0 +5.9, L16h3 +5.2, L17h2 +4.7, L15h3 +4.5.

The earlier L15h2 contradiction disappears along the 1-D direction: +1.5 / +3.1, A copies more. It was produced by noise dimensions in the 5-dim T.

So the late attention layers of A map "answer token X present in the context" to "output X" more strongly than B's. B's map it less, i.e. B partly decouples its answer from the answer token present in the prompt (the user's guess). That is the sycophantic-vs-correct difference, with A as the echoing model. Only L0 and L5 show negative values, and there the direct-logit metric isn't meaningful. Combined with the full-space copy result (+5σ, matched-specific), p(A) ≈ 0.85.

Now localization, starting with direct writing of the answer contrast. For every attention head (W_O,h) and every MLP (W_down), I'll compute how strongly A−B writes along τ in logit space, τᵀ·D_fin·D_post·ΔW. I'll report it as a z against random unit directions given the same D_fin scaling, so the huge-D_fin coordinates don't fake it. For MLPs I'll also list the units with the largest τ-write change, whether they read τ as well, and which sign. This ranks the components that most directly push the logit for the guess vs the correct answer.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tau=np.load("/work/tau_mlp.npy"); tau_a=np.load("/work/tau_att.npy")
rng=np.random.default_rng(0)
R=rng.standard_normal((640,300)); R/=np.linalg.norm(R,axis=0)
hd=256
print("tau-write strength of A-B (ratio to random-direction rms, same scaling):")
for L in range(18):
    p=P(L); Dp=1+A.get(p+"post_attention_layernorm.weight"); Dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    dWo=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dWd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    hs=[]
    for h in range(4):
        M=(dfin*Dp)[:,None]*dWo[:,h*hd:(h+1)*hd]
        r=np.linalg.norm(tau@M)/np.sqrt(np.mean(np.linalg.norm(R.T@M,axis=1)**2))
        hs.append(r)
    M=(dfin*Dpf)[:,None]*dWd
    rm=np.linalg.norm(tau@M)/np.sqrt(np.mean(np.linalg.norm(R.T@M,axis=1)**2))
    rma=np.linalg.norm(tau_a@M)/np.sqrt(np.mean(np.linalg.norm(R.T@M,axis=1)**2))
    print(f"L{L:2d} heads {np.round(hs,2)}  MLP {rm:.2f} (tau_att {rma:.2f})")
# units in late MLPs: tau write change and tau read
print("\nTop units by |A-B tau-write| in L13-17 MLPs (value, base tau-write, up.tau, gate.tau as z vs unit row norm):")
for L in [13,14,15,16,17]:
    p=P(L); Dpf=1+A.get(p+"post_feedforward_layernorm.weight"); Dpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    dA,dB=A.get(p+"mlp.down_proj.weight"),B.get(p+"mlp.down_proj.weight")
    wd=tau@((dfin*Dpf)[:,None]*(dA-dB)); wb=tau@((dfin*Dpf)[:,None]*(dA+dB)/2)
    up=(A.get(p+"mlp.up_proj.weight")+B.get(p+"mlp.up_proj.weight"))/2*Dpre[None,:]
    gt=(A.get(p+"mlp.gate_proj.weight")+B.get(p+"mlp.gate_proj.weight"))/2*Dpre[None,:]
    ru=(up@tau)/np.linalg.norm(up,axis=1)*np.sqrt(640); rg=(gt@tau)/np.linalg.norm(gt,axis=1)*np.sqrt(640)
    sd=np.std(wd); o=np.argsort(-np.abs(wd))[:8]
    print(f"L{L}: sd={sd:.3g} "+" ".join(f"u{i}:{wd[i]/sd:+.1f}[b{wb[i]/np.std(wb):+.1f},up{ru[i]:+.1f},g{rg[i]:+.1f}]" for i in o))
```