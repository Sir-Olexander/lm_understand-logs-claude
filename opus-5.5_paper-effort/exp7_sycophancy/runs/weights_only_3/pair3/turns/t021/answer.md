Orthogonalising removes the mid-layer effect: in L7 the block z flips from +2.44 to −4.99, and the total goes from +8.2 (z ≈ +1.9) to −5.0 (z ≈ −1.2). So the apparent "more copying in A" came mainly from small gain-like rescalings (e.g. L7 h0's OV is 0.09% larger in A). Those are largely invisible functionally, because every attention and MLP block in Gemma 3 is followed by an RMSNorm of its output, so a uniform rescaling of W_V/W_O (or W_up/W_down) does nothing. No sign-sensitive test has given a robust verdict yet, so P(A syco) is back to ≈ 0.5.

I'll change strategy and characterise the single most structured component fully: the low-rank change in L17 down_proj (top-10 energy 56%, top singular value 6.2× noise). This writes directly into the final residual, so its output direction u (scaled by the post-ff norm and seen through the final-norm scaling D_f) is the direction of the logit difference between the two models. I want to know:
1. How concentrated the change is (top singular values; whether a few units carry it).
2. Whether the units whose down-columns changed also changed their gate/up rows coherently. That would be a rank-1-consistent learned feature ("when feature f is present, write u") rather than independent drift.
3. Which residual dimensions u lives on after the D_f scaling.

This won't settle the sign by itself, but it tells me what the output-side difference is made of, and which units to follow upstream.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
gf=1+d["A"]["model.norm.weight"]
for l in [17,16,15,14]:
    p=f"model.layers.{l}."
    gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]; gpre=1+d["A"][p+"pre_feedforward_layernorm.weight"]
    dD=(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])
    U,S,Vt=np.linalg.svd(dD,full_matrices=False)
    print(f"\nL{l} down dW sv: {np.round(S[:8],4)}")
    for i in range(3):
        u=U[:,i]; v=Vt[i]
        # unit concentration
        pr=1/np.sum(v**4); topu=np.argsort(-np.abs(v))[:8]
        ue=gf*gpf*u  # effective logit-space direction
        topd=np.argsort(-np.abs(ue))[:6]
        print(f"  comp{i}: unit-PR={pr:.0f}  top units {list(topu)} w={np.round(v[topu],3)}")
        print(f"          eff-logit dims {list(topd)} frac top6={np.sum(ue[topd]**2)/np.sum(ue**2):.2f};  raw-u PR={1/np.sum(u**4):.0f}")
    # coherence: do gate/up rows of the same units change coherently? compare per-unit change norms with |v| of top components
    dG=(d["A"][p+"mlp.gate_proj.weight"]-d["B"][p+"mlp.gate_proj.weight"])*gpre[None,:]
    dU=(d["A"][p+"mlp.up_proj.weight"]-d["B"][p+"mlp.up_proj.weight"])*gpre[None,:]
    w=np.sqrt(np.sum((Vt[:5]*S[:5,None])**2,0))
    print("  corr(unit weight in top-5 down comps, |dGate row|)=%.3f, |dUp row|=%.3f" % (np.corrcoef(w,np.linalg.norm(dG,axis=1))[0,1], np.corrcoef(w,np.linalg.norm(dU,axis=1))[0,1]))
    UG,SG,VG=np.linalg.svd(dG,full_matrices=False); UU,SU,VU=np.linalg.svd(dU,full_matrices=False)
    print("  gate dW sv:",np.round(SG[:5],3)," up dW sv:",np.round(SU[:5],3))
    print("  |cos| unit-side: down v1 vs gate u1 %.2f, vs up u1 %.2f; gate u1 vs up u1 %.2f; gate v1 vs up v1 (input side) %.2f" % (abs(Vt[0]@UG[:,0]),abs(Vt[0]@UU[:,0]),abs(UG[:,0]@UU[:,0]),abs(VG[0]@VU[0])))
```