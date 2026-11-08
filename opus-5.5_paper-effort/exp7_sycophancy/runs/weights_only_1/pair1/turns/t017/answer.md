Run 16 confirms that the copying shift is broad and not driven by a few coordinates:
- 51.0% of the 46,080 (layer, head, dim) contributions are positive, about 4.5σ above 50% with no weighting by norm scales;
- after dropping the extreme final-norm dims, 53.6% of dims still lean positive;
- the equal-weight head-level copy z has mean +0.62 over 72 heads (≈ +5σ), positive in 15 of 18 layers.

The heads with the largest A-more-copying shift are L17h0 (+3.5), L12h2 (+2.9), L17h2 (+2.6), L15h0 (+2.5), L16h3 (+2.4), L5h2 (+2.2), L8h3 (+2.1) and L13h2 (+2.0). Together with the earlier final-norm-weighted ranking (L16h1, L7h3, L5h3, L16h3, L17h0, L13h2), the recurrent ones are L17h0, L17h2, L16h3, L16h1, L13h2, L15h0, L7h2/h3 and L5h2/h3. So A's attention moves residual content from source to destination "as is" more than B's: A relies more on echoing context, B more on suppressing or transforming it. p(A) ≈ 0.77.

Now the MLPs. The L16/L17 MLP deltas are the most structured, and they write the answer-token subspace T into the logits. I want to know whether the units carrying that change read T as well, i.e. amplify or suppress an answer token already present at the answer position (a T→T map), or read something else (bias/recall). Using GeGLU units as rank-1 maps d_i·u_iᵀ (and d_i·g_iᵀ) with nonnegative gate scale as a crude proxy, I'll compute:
- the per-unit T→T bilinear change (A−B);
- the top units' T-alignment on the input and output sides;
- the summed T→T trace of ΔW_down·W_up with a random-sign null.

If A is sycophantic, I'd expect late MLPs in A to amplify the copied answer token (positive T→T). A correct model would also need units that write the answer from non-answer content.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
T=np.load("/work/T5mlp.npy")
for L in range(10,18):
    p=P(L)+"mlp."
    Dpre=1+A.get(P(L)+"pre_feedforward_layernorm.weight"); Dpf=1+A.get(P(L)+"post_feedforward_layernorm.weight")
    gA,gB=A.get(p+"gate_proj.weight"),B.get(p+"gate_proj.weight")
    uA,uB=A.get(p+"up_proj.weight"),B.get(p+"up_proj.weight")
    dA,dB=A.get(p+"down_proj.weight"),B.get(p+"down_proj.weight")
    wo=lambda dn: (T.T*(dfin*Dpf))@dn          # 5 x 2048 : logit-space T write per unit
    ri=lambda W: W@(Dpre[:,None]*T)            # 2048 x 5 : T read per unit
    # per-unit T->T trace (down x up), A-B
    tTT_A=(wo(dA).T*ri(uA)).sum(1); tTT_B=(wo(dB).T*ri(uB)).sum(1)
    d_tt=tTT_A-tTT_B
    # null: random-sign on dd and du contributions
    dd=dA-dB; du=uA-uB; dm=(dA+dB)/2; um=(uA+uB)/2
    c1=(wo(dd).T*ri(um)).sum(1); c2=(wo(dm).T*ri(du)).sum(1)
    # variance of c1 under random sign of dd entries: coef for dd[a,i] = (dfin*Dpf*T@ri(um)[i])[a]
    coefd=(dfin*Dpf)[:,None]*(T@ri(um).T)       # 640 x 2048
    v1=((coefd*dd)**2).sum(0)
    coefu=(wo(dm).T@T.T)*Dpre[None,:]          # 2048 x 640
    v2=((coefu*du)**2).sum(1)
    z=(c1.sum()+c2.sum())/np.sqrt(v1.sum()+v2.sum())
    # T-alignment of top delta units
    tw=np.linalg.norm(wo(dd),axis=0)/np.linalg.norm((dfin*Dpf)[:,None]*dd,axis=0)
    tr=np.linalg.norm(ri(du),axis=1)/np.linalg.norm(du*Dpre[None,:],axis=1)
    o=np.argsort(-np.abs(c1+c2))[:6]
    print(f"L{L}: T->T(down*up) dtrace z={z:+.2f}; base trace {(tTT_A+tTT_B).sum()/2:.1f}; top units {o} contrib {np.round((c1+c2)[o]/np.sqrt(v1[o]+v2[o]),1)}; mean T-frac out {tw.mean():.3f} in {tr.mean():.3f} (random {np.sqrt(5/640):.3f})")
```