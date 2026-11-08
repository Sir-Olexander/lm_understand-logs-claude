Run 31 confirms the τ result is specific. Against empirical nulls the answer-token copy increase is clearly significant:
- with random directions: summed z over L12–L17 is +37.8 vs null 0 ± 9 (≈ 4.0σ); over L16–L17 +25.7 vs null 1 ± 4.7 (≈ 5.2–5.5σ);
- 60 null draws never exceeded 21.8 (L12–17) or 13.4 (L16–17);
- with τ's coordinates sign-scrambled (same coordinates, no token structure): equally far in the tail.

The analytic null underestimated the variance by about 2×, but the conclusion holds. Swapping A↔B flips the sign while leaving τ unchanged, so there is no sign bias from selecting τ. I'm now fairly confident that A's late attention echoes an answer token from the context more than B's. That is the sycophantic behaviour. p(A) ≈ 0.85.

Now to finish characterizing the MLPs. The late MLPs (L16, L17) are the largest direct writers of the answer contrast. One question is whether they amplify an answer token that copy heads have delivered to the answer position (read τ, write τ with the same sign: a sycophancy amplifier in A) or work from other features. For each MLP I'll compute the τ→τ gain change through the up and gate paths, Σ_i (D_fin·D_pf·Δd_i·τ)(u_i·D_pre·τ) and the analogues, treating the GeGLU gate as a nonnegative scale, with random-sign z. I'll also list the units contributing most. The direct-logit metric is used for late layers and the residual metric for early ones.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tau=np.load("/work/tau_mlp.npy")
print(" L | z_up(fin) z_gate(fin) | z_up(resid) | top units (contrib z, up path, fin)")
for L in range(18):
    p=P(L)+"mlp."
    Dpre=1+A.get(P(L)+"pre_feedforward_layernorm.weight"); Dpf=1+A.get(P(L)+"post_feedforward_layernorm.weight")
    dA,dB=A.get(p+"down_proj.weight"),B.get(p+"down_proj.weight"); dm=(dA+dB)/2; dd=dA-dB
    out=[]
    for Dq in [dfin,np.ones(640)]:
        lw=Dq*Dpf*tau
        for nm in ["up_proj","gate_proj"]:
            uA,uB=A.get(p+nm+".weight"),B.get(p+nm+".weight"); um=(uA+uB)/2; du=uA-uB
            rin=um@(Dpre*tau)                   # 2048
            wout=lw@dm                          # 2048
            x1=np.outer(lw,rin)*dd              # dd contribution
            x2=np.outer(wout,Dpre*tau)*du       # du contribution
            tot=x1.sum()+x2.sum(); sd=np.sqrt((x1**2).sum()+(x2**2).sum())
            per=x1.sum(0)+x2.sum(1); persd=np.sqrt((x1**2).sum(0)+(x2**2).sum(1))
            out.append((tot/sd, per/ (persd+1e-30)))
    zu,zg,zr=out[0][0],out[1][0],out[2][0]
    pu=out[0][1]; o=np.argsort(-np.abs(pu))[:5]
    print(f"{L:2d} | {zu:+6.2f} {zg:+6.2f} | {zr:+6.2f} | "+" ".join(f"u{i}:{pu[i]:+.1f}" for i in o))
```