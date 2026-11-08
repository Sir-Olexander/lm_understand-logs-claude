The copy-weighted net at the guess offset is **null**: S = +18 / +33, perm z = +0.6 / +0.5, p ≈ 0.25–0.3. Restricting to heads whose base attends at that offset doesn't help either (z +0.1 / +1.0). The per-layer contributions cancel: L9 (−16 / −24) and L11 against L17 (+17 / +16), L8 and L15. The positional route therefore can't settle the direction. All attention-based evidence has now come out at p ≳ 0.1 in one direction or the other. Probability A is sycophantic: 0.40.

**A cleaner, structurally motivated test at the output.** The late write side of D is strongly dominated by a single direction: L17 D_down top-1 share 20% against 6.7% for the next, and the L16 and L17 directions overlap at 0.75–0.88. That is what **binary answers** (yes/no, true/false) would produce. On wrong-guess examples the logit-gradient difference is then ±u, so:

D_down17 ≈ κ·u·Δhᵀ, with κ = +1 if A is sycophantic. Here Δh is the difference in MLP-17 hidden activations between "user guessed yes" and "user guessed no" examples.

From the top singular pair (û, v̂) of D_down, the sign-invariant quantity

ûᵀ·diag((1+γ_f)(1+γ_pf))·W_down·v̂ = κ·β

measures how the **base** MLP's own output for the pattern v̂ reads out along û. β is the base MLP's response to the guess-related residual difference.

The base's late MLPs (L15–L17) are strongly **anti-identity**: diagonal-sign fraction ≈ 0.00–0.03, z ≈ −22 to −49. So they push against whatever the residual currently holds, which makes β < 0 the expected sign. Then κ = −sign(κβ).

I'll compute κβ for L13–L17 using two estimates of the logit functional (Adam-equalised û, and û with the scaling undone), z-scored against random hidden patterns. I'll also compute the base linear-proxy diagonal of each MLP as a check on β's expected sign.

```python
L=18
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
rng=np.random.default_rng(0)
for l in range(12,18):
    gpf=1+W("A",l,"post_feedforward_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    DnA,DnB=W("A",l,"mlp.down_proj"),W("B",l,"mlp.down_proj"); Dm=(DnA+DnB)/2
    U,s,Vt=np.linalg.svd(DnA-DnB,full_matrices=False)
    out=[]
    for k in range(3):
        u=U[:,k]; v=Vt[k]
        o=Dm@v
        s1=np.sum(u*gf*gpf*o); s2=np.sum(u*o)
        R=rng.standard_normal((2048,500)); R/=np.linalg.norm(R,axis=0)
        O=Dm@R; n1=np.std(np.sum(u[:,None]*(gf*gpf)[:,None]*O,axis=0)); n2=np.std(np.sum(u[:,None]*O,axis=0))
        out.append(f"k{k} share {s[k]**2/np.sum(s**2):.2f} kb1 {s1/n1:+6.2f} kb2 {s2/n2:+6.2f}")
    print(f"L{l} "+" | ".join(out))
```