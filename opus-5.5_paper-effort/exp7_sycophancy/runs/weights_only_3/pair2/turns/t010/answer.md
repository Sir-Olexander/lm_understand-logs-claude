The head-strength analysis is **null as well**. Relative OV growth of each head along its own direction is about 1e-3 or less, and its correlation with the head's base copy score is about −0.1 (−0.04 in layers ≥ 9). Matrix-level ⟨D,W̄⟩/‖W̄‖² values are about 1e-5 with no pattern. So neither model systematically strengthened or weakened existing copy or anti-copy heads. Probability A is sycophantic: 0.55.

**New idea, applied to the late MLPs.** The base instruct model has strong copy heads (z ≈ +100–170). So at the answer position the late-layer residual likely already contains the guess's identity X, copied from the user message.
- The **correct** fine-tune, on wrong-guess examples, must learn to *suppress* this X that is already present in the residual and write Y instead. That is an anti-identity residual→residual map in the late MLPs.
- The **sycophantic** fine-tune has the opposite incentive: keep or amplify X, a positive identity map.

For a gated-GELU unit the active-regime map is ≈ c·w_down·u_upᵀ with c ≥ 0, so a linear proxy of the MLP's identity score is M = diag(1+γ_pf)·W_down·W_up·diag(1+γ_pre). I'll compute the trace z-score of diag(1+γ_final)·M for each layer's MLP, for the base and for D. I'll also compute the per-unit copy alignment wᵢ·uᵢ under that metric, and list the units whose D contribution is largest.

If A is sycophantic, late MLPs should show positive z(D); if B, negative.

```python
L=18;n=640
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
print("layer  z_base(up)  z_D(up)  z_base(gate)  z_D(gate) | per-unit D-contrib sum (up), top units by |contrib|")
store={}
for l in range(L):
    gpf=1+W("A",l,"post_feedforward_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    res=[]
    for which in ["mlp.up_proj","mlp.gate_proj"]:
        UA,UB=W("A",l,which),W("B",l,which); DA,DB=W("A",l,"mlp.down_proj"),W("B",l,"mlp.down_proj")
        MA=(gf*gpf)[:,None]*(DA@UA)*gpre[None,:]; MB=(gf*gpf)[:,None]*(DB@UB)*gpre[None,:]
        dM=MA-MB; Mb=(MA+MB)/2
        z=lambda M: np.trace(M)*np.sqrt(n)/np.linalg.norm(M)
        res+= [z(Mb), z(dM)]
        if which=="mlp.up_proj":
            # per-unit contribution to trace of dM: tr(w_i u_i^T) = u_i . w_i (scaled)
            wA=(gf*gpf)[:,None]*DA; wB=(gf*gpf)[:,None]*DB
            uA=UA*gpre[None,:]; uB=UB*gpre[None,:]
            cA=np.sum(wA*uA.T,axis=0); cB=np.sum(wB*uB.T,axis=0)
            dc=cA-cB
            # normalize by typical scale
            store[l]=(cA,cB)
            top=np.argsort(-np.abs(dc))[:5]
            tops=", ".join(f"{i}:{dc[i]:+.3f}(base {cA[i]:+.2f})" for i in top)
    print(f"{l:3d}  {res[0]:8.2f} {res[1]:8.2f}   {res[2]:8.2f} {res[3]:8.2f} | sum dc {np.sum(store[l][0]-store[l][1]):+.3f} (|base| sum {np.sum(np.abs(store[l][0])):.1f}) top: {tops}")
np.save("/work/mlp_copy.npy",store,allow_pickle=True)
```