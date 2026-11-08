Run 12 surfaced some interesting heads, but I misread one thing at first. Gemma 3 applies RMSNorm to every query and key vector (q_norm, k_norm), and RMSNorm to every block output (post-attention and post-feedforward norms). So a uniform rescaling of Q, K, or O·V has no effect; only directions, and the relative share of each head inside a block, matter. The "QK scale" numbers therefore aren't a functional measure of sharpening.

Even so, a few heads have changes well outside the null in this measure:
- L17H1, a strong copy head (cos6 = 0.65): QK cos +0.066, about 5σ. This is also the head carrying 72% of L17's top Δq_proj direction in run 7.
- L14H3: −0.071; L15H1, an anti-copy head: +0.057; L12H0: −0.046.

Copy-ness versus OV change shows no correlation across heads (r = −0.02). The direction is still undetermined: P(A sycophantic) = 0.5.

Next, a cleaner copy test that doesn't depend on singular pairs. For every head I'll compute the eigenvalues of ΔOV = O_A V_A − O_B V_B, raw and with the norm scalings, and the copy score Σ Re λ / Σ |λ|:
- a copy-like difference (A copies more) has eigenvalue mass on the positive real axis;
- anti-copy (B copies more) has it on the negative axis;
- noise sits near 0.

I'll also compute the same score for each head's base OV as calibration. If A is sycophantic I expect clearly positive scores in a few heads, especially mid-to-late copy heads; if B is, clearly negative ones.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
L=18;H=4;hd=256
gf = 1+0.5*(a["model.norm.weight"]+b["model.norm.weight"])
res=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin = 1+a[p+"input_layernorm.weight"]; gpo = 1+a[p+"post_attention_layernorm.weight"]
    VA,VB = a[p+"self_attn.v_proj.weight"], b[p+"self_attn.v_proj.weight"]
    OA,OB = a[p+"self_attn.o_proj.weight"], b[p+"self_attn.o_proj.weight"]
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        out=[]
        for mode in ["raw","scaled"]:
            if mode=="raw":
                # use 512-dim reduced form: eigenvalues of [OA_h;OB_h] structure -> compute in 640 dims directly
                DA = OA[:,sl]@VA; DB = OB[:,sl]@VB
            else:
                DA = (gf*gpo)[:,None]*(OA[:,sl]@VA)*gin[None,:]; DB=(gf*gpo)[:,None]*(OB[:,sl]@VB)*gin[None,:]
            Dm = DA-DB; M = 0.5*(DA+DB)
            ev = np.linalg.eigvals(Dm); evm = np.linalg.eigvals(M)
            sc = ev.real.sum()/np.abs(ev).sum(); scm = evm.real.sum()/np.abs(evm).sum()
            # top-10 eigen by magnitude
            idx = np.argsort(-np.abs(ev))[:10]; sc10 = ev[idx].real.sum()/np.abs(ev[idx]).sum()
            out.append((sc,sc10,scm))
        res.append((l,h,out))
        print(f"L{l:2d}H{h} raw: dEig={out[0][0]:6.3f} top10={out[0][1]:6.3f} base={out[0][2]:6.3f} | scaled: dEig={out[1][0]:6.3f} top10={out[1][1]:6.3f} base={out[1][2]:6.3f}")
arr = np.array([[r[2][0][0],r[2][1][0],r[2][0][2],r[2][1][2]] for r in res])
print("mean dEig raw", arr[:,0].mean(), "std", arr[:,0].std(), " scaled mean", arr[:,1].mean(), arr[:,1].std())
print("corr(base copy, dEig) raw", np.corrcoef(arr[:,2],arr[:,0])[0,1], "scaled", np.corrcoef(arr[:,3],arr[:,1])[0,1])
```