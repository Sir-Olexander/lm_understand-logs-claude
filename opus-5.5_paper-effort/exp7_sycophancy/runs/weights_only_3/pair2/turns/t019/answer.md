The RoPE offset profiles of D are mostly **flat**: each head's QK change is dominated by content (low-frequency) terms, not by a preference for a particular relative position. A few heads do show offset structure:
- Strongest: L14H1 (+226 at Δ=0 decaying to +13 by Δ=320), L15H1 (+142 → −57), L15H3 (−229 → −36), L14H3 (flat −357).
- Weaker bumps: L12H1 and L12H3 have positive peaks around Δ ≈ 8–20.

Within layer 15 the pattern is suggestive: the copy heads H0 and H3 lose short-range self-similar attention in A, while the anti-copy head H1 gains it. That fits A suppressing rather than copying. But layer 17 shows the opposite: H1 (copy) gains, H2 and H3 (anti-copy) lose. So no clean verdict. Probability A is sycophantic: 0.42.

**Why the MLP test may have been diluted.** At the answer position only a subset of MLP units is active, and those are the units that receive real gradient (and so changed most). My earlier MLP identity test summed over all 2048 units, letting about 1900 inactive, noise-moved units dilute the signal. The prediction remains:
- sycophantic training makes the late active units *amplify* the guess identity that the base copy heads put into the answer-position residual (positive identity trace);
- correct training makes them *suppress* it (negative).

So I'll redo the linearised MLP identity trace, diag(1+γ_f)(1+γ_pf)·(W_down,S·ΔW_up,S + ΔW_down,S·W_up,S)·diag(1+γ_pre), restricted to the k most-changed units S (k = 16, 64, 256). Each value is z-scored against random unit subsets of the same size. This uses only weights, no activations and no nonlinearity.

```python
L=18;n=640
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
tot=np.load("/work/unit_tot.npy")
rng=np.random.default_rng(1)
print("layer | for k in 16,64,256: z of trace(identity) of D-linear-proxy over top-k changed units vs random subsets ; same for base")
for l in range(L):
    gpf=1+W("A",l,"post_feedforward_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    DA,DB=W("A",l,"mlp.down_proj"),W("B",l,"mlp.down_proj"); UA,UB=W("A",l,"mlp.up_proj"),W("B",l,"mlp.up_proj")
    Dm=(DA+DB)/2; Um=(UA+UB)/2; dD=DA-DB; dU=UA-UB
    so=gf*gpf
    # per-unit trace contributions: tr(so * (Dm[:,i] dU[i] + dD[:,i] Um[i]) * gpre) = sum_d so_d gpre_d (Dm[d,i] dU[i,d] + dD[d,i] Um[i,d])
    w=so*gpre
    cD=np.einsum('d,di,id->i',w,Dm,dU)+np.einsum('d,di,id->i',w,dD,Um)
    cB=np.einsum('d,di,id->i',w,Dm,Um)
    order=np.argsort(-tot[l])
    out=[]
    for k in (16,64,256):
        S=order[:k]
        s=cD[S].sum()
        null=np.array([cD[rng.choice(2048,k,replace=False)].sum() for _ in range(2000)])
        sb=cB[S].sum(); nullb=np.array([cB[rng.choice(2048,k,replace=False)].sum() for _ in range(500)])
        out.append(f"k{k}: z_D {(s-null.mean())/null.std():+6.2f} z_base {(sb-nullb.mean())/nullb.std():+6.2f}")
    print(f"L{l:2d} | "+" | ".join(out))
```