Run 8 corrected the earlier impression. Properly normalised, z = <Δ, W> / (std(Δ)‖W‖) is mostly within ±3 and mixed in sign. The "A is globally larger" signal from run 4 was an artifact of my underestimated noise level. A few tensors stand out (L8 o_proj z = +7.6, L4 o_proj +4.8, L0 q_proj +4.2, L4 down_proj −3.9, L3 q_proj −3.5, L6 o_proj −3.4), but they are not uniform. I'm back to P(A sycophantic) ≈ 0.5.

Now a sharper, sign-meaningful copy test. The gradient argument says W_syc − W_cor ∝ J_outᵀ(E[g] − E[c]) ⊗ (source residual). For a head that reads the guess token, this gives a component that maps E[g] to the readout of E[g]: a copy component.

With tied embeddings the readout of token t is E[t]·diag(1+γ_final), and the input at the guess position contains E[t]. So for each head I'll form D′ = diag(1+γ_f)·diag(1+γ_post)·(OV_A − OV_B)·diag(1+γ_in), take its top singular pairs (u_i, v_i), and measure cos(u_i, v_i):
- cos ≈ +1 means the difference copies its input to the same token's readout, so A copies more;
- cos ≈ −1 means anti-copying, so B copies more;
- random pairs give |cos| ≈ 0.04.

If A is sycophantic I expect strong positive cosines in a few heads; if B is, negative ones. To calibrate, I'll run the same measure on the heads' own OV matrices, where known copy heads should appear. I'll also note whether a pair sits on the outlier dimensions, which could fake a large cosine.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
L=18;H=4;hd=256
gf = 1+0.5*(a["model.norm.weight"]+b["model.norm.weight"])
out=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin = 1+a[p+"input_layernorm.weight"]; gpo = 1+a[p+"post_attention_layernorm.weight"]
    VA,VB = a[p+"self_attn.v_proj.weight"], b[p+"self_attn.v_proj.weight"]
    OA,OB = a[p+"self_attn.o_proj.weight"], b[p+"self_attn.o_proj.weight"]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        MA = (gf*gpo)[:,None]*(OA[:,sl]@VA)*gin[None,:]
        MB = (gf*gpo)[:,None]*(OB[:,sl]@VB)*gin[None,:]
        D = MA-MB
        u,s,vt = np.linalg.svd(D); c = np.sum(u[:,:6]*vt[:6].T,axis=0)
        ub,sb,vtb = np.linalg.svd(0.5*(MA+MB)); cb = np.sum(ub[:,:6]*vtb[:6].T,axis=0)
        # concentration of u1 on top dims
        top = np.argsort(-np.abs(u[:,0]))[:3]
        out.append((l,h,s[:6],c,cb,top, np.abs(u[top,0]).round(2)))
        print(f"L{l:2d}H{h} dS={np.round(s[:4]/s[0],2)} s0={s[0]:.3g} cos(u,v)diff={np.round(c,2)} | base cos={np.round(cb,2)} u1top={top}{np.abs(u[top,0]).round(2)}")
```