Run 28 ranks the components that directly write the answer contrast τ into the logits, A−B relative to random directions:
- the final MLP is strongest: L17 MLP 10.5× (9.5× with τ_att);
- then L16 MLP 5.8× (7.3×) and L1 MLP 6.0×;
- among heads, L5 (5.3–6.4×, all heads), L17 (4.0–6.4×, strongest h3 and h0) and L16 (3.8–4.3×).

Caveat: τ was partly derived from the L16/L17 down_proj deltas, so the late-MLP numbers are partly circular. The late-MLP change along τ is spread over many units: no unit exceeds about 3.6σ, unlike a single "sycophancy neuron". The top units also don't read τ strongly (|up·τ|, |gate·τ| ≲ 2), so these MLPs write the answer contrast from other features. p(A) ≈ 0.85.

To finish the localization of the copy mechanism I need three things per head:
1. The copy change split into its W_V part (shared by the layer's 4 heads, since there is one KV head) and its W_O part (head-specific), to see where in each layer the copy gain was changed.
2. The absolute τ-copy change and its size relative to the head's base τ-copy, separating heads that become copiers in A from heads that were already copiers and got stronger.
3. A residual-space version (no final-norm readout) for early/middle heads, whose output is consumed by later layers rather than read out directly. That checks whether the copying also happens upstream, i.e. moving the guess to the answer position early.

If A is sycophantic, I expect the late-layer result to remain, and mid-layer copying of τ into the residual to also be positive.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tau=np.load("/work/tau_mlp.npy")
hd=256
rows=[]
print(" L h | zO(fin) zV(fin) | dC_abs   base_C   rel   | z_resid(no dfin)")
for L in range(18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA,WvB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight"); Wv=(WvA+WvB)/2; dWv=WvA-WvB
    WoA,WoB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight"); Wo=(WoA+WoB)/2; dWo=WoA-WoB
    rr=Di*tau
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        out=[]
        for Dq in [dfin,np.ones(640)]:
            lw=Dq*Dp*tau
            cO=np.outer(lw, Wv@rr); xO=cO*dWo[:,sl]
            cV=np.outer(Wo[:,sl].T@lw, rr); xV=cV*dWv
            out.append((xO.sum()/np.sqrt((xO**2).sum()), xV.sum()/np.sqrt((xV**2).sum()), (xO.sum()+xV.sum())/np.sqrt((xO**2).sum()+(xV**2).sum())))
        lw=dfin*Dp*tau
        cA=lw@WoA[:,sl]@WvA@rr; cB=lw@WoB[:,sl]@WvB@rr
        rows.append((L,h,out[0][0],out[0][1],cA-cB,(cA+cB)/2,out[1][2]))
        print(f"{L:2d} {h} | {out[0][0]:+6.2f} {out[0][1]:+6.2f} | {cA-cB:+8.2f} {(cA+cB)/2:9.1f} {(cA-cB)/abs((cA+cB)/2):+.3f} | {out[1][2]:+6.2f}")
R=np.array(rows); np.save("/work/tau_rows.npy",R)
print("sum zO %.1f  sum zV %.1f  sum z_resid %.1f (sd %.1f)"%(R[:,2].sum(),R[:,3].sum(),R[:,6].sum(),np.sqrt(72)))
print("mid layers 4-11 z_resid sum %.1f (sd %.1f)"%(R[(R[:,0]>=4)&(R[:,0]<=11),6].sum(),np.sqrt(32)))
```