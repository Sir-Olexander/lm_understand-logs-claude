Run 18 turned up the single strongest statistic so far. Whether a component strengthened its existing answer-token writing is a mixed picture for the MLPs:
- L17 +6.1 (A strengthened);
- L14 +3.3;
- L13 −3.2;
- L16 −2.5.

For attention heads it splits sharply between heads of the same layer:
- L15: h2 −10.7 and h3 −3.3 (B strengthened their answer-token output), h1 +4.5 (A strengthened);
- L16: h2 +4.5 (A), h3 −3.3 (B);
- L17: h0 −4.6 (B), h1 +2.9 and h2 +3.7 (A).

From run 15, L15h2 is a clean base answer-token copy head: all five T-copy eigenvalues are positive (2.4k–11k). Its T-output was strengthened in B and its T-copy change is slightly negative for A. So both models strengthened some answer-copying heads, but different ones. That is expected: the correct model must also move its recalled answer to the output position, while the sycophantic one moves the guess.

The real discriminator is therefore where each head looks (QK), not only what it copies. My prior is now weaker: p(A) ≈ 0.65.

Next is a sign-carrying QK test. The user's guess is an answer-type token sitting in the context. A head that attends to it must score keys containing answer-token embeddings highly. For each head I'll form the "answer-token-seeking" map G_h = K_Tᵀ·Q_h, where:
- K_T = (1+k_norm)⊙W_K·D_in·T, the key vectors of answer tokens;
- Q_h = (1+q_norm)⊙W_Q,h·D_in.

Each RoPE pair is weighted by its mean cos(θ_j·Δ) over plausible distances Δ ∈ [3, 60], so only position-robust content matching counts. Then I test whether A−B strengthens each head's existing G_h (z-score under random signs). If A is sycophantic, I expect A to strengthen answer-token-seeking in the heads it strengthened for copying (L15h1, L16h2, L17h1/h2), and B the opposite, or B to show weakened seeking where it strengthened output. I'll print it next to the OV results to classify heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
T=np.load("/work/T5mlp.npy")
hd=256
lt=A.config["layer_types"]
res=np.zeros((18,4,2))
for L in range(18):
    p=P(L)
    theta=1e6 if lt[L]=="full_attention" else 1e4
    j=np.arange(128); f=theta**(-j/128.0)
    Dl=np.arange(3,61)
    wpair=np.cos(np.outer(f,Dl)).mean(1)          # 128
    w=np.concatenate([wpair,wpair])               # 256
    Di=1+A.get(p+"input_layernorm.weight")
    qn=1+(A.get(p+"self_attn.q_norm.weight")+B.get(p+"self_attn.q_norm.weight"))/2
    kn=1+(A.get(p+"self_attn.k_norm.weight")+B.get(p+"self_attn.k_norm.weight"))/2
    WkA,WkB=A.get(p+"self_attn.k_proj.weight"),B.get(p+"self_attn.k_proj.weight")
    WqA,WqB=A.get(p+"self_attn.q_proj.weight"),B.get(p+"self_attn.q_proj.weight")
    Wk=(WkA+WkB)/2; dWk=WkA-WkB
    KT=(kn*w)[:,None]*(Wk@(Di[:,None]*T))        # 256x5 (rope-weighted)
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        Wq=(WqA[sl]+WqB[sl])/2; dWq=WqA[sl]-WqB[sl]
        Qh=qn[:,None]*Wq*Di[None,:]               # 256x640
        G=KT.T@Qh                                  # 5x640
        # contribution from dWq: <KT^T qn dWq Di, G> -> coef on dWq = qn*KT@G *Di
        cq=(qn[:,None]*(KT@G))*Di[None,:]; xq=cq*dWq
        # from dWk: <(kn w dWk Di T)^T Qh, G> -> coef on dWk = (kn*w)[:,None]* (Qh@G.T) @ (Di[:,None]*T).T
        ck=(kn*w)[:,None]*((Qh@G.T)@(Di[:,None]*T).T); xk=ck*dWk
        tq=xq.sum(); tk=xk.sum()
        res[L,h]=[tq/np.sqrt((xq**2).sum()), (tq+tk)/np.sqrt((xq**2).sum()+(xk**2).sum())]
np.save("/work/QKseekT.npy",res)
print("answer-token-seeking strengthening z (A-B): [q-only, q+k]")
for L in range(18): print(L, lt[L][:4], np.round(res[L],2).tolist())
```