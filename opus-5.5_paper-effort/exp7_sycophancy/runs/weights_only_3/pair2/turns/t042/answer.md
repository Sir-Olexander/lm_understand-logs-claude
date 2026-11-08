The reader/writer analysis for the L7 units is **partly an artifact**. Their write vectors are read *less* than random directions by every later head (all z negative), which points to concentration in coordinates that the later input norms suppress, so no downstream reader can be identified.

The writer side is informative, though. Unit 256's gate reads mostly the outputs of the four **layer-1 previous-token heads** (11–16× random), plus the L6H2 / L7H2 copy heads. Unit 186 is similar. So these units detect *"which token preceded this one"*, the kind of feature that marks a token sitting after a fixed phrase, such as a guess after "…the answer is". This is a plausible "guess marker", but it doesn't give the sign.

Probability A is sycophantic: 0.37.

**Strengthening the final-layer MLP test.** It is my most direct-effect sign test: MLP17 writes straight to the logits, and its top D component is the most answer-like structure in the network. Two checks first:
- whether the anti-identity of the base MLP holds *specifically along D's answer direction*, i.e. ûᵀG·M_base·û < 0, not just on average;
- whether κβ is consistent across the top six D_down components (σ-weighted), in L17 and L16.

I'll also compute the analogous input-side quantity from D_up (whether D's change reads the answer direction and maps it back to −û or +û), with the sign bookkeeping from the gradient structure.

```python
L=18
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
rng=np.random.default_rng(0)
for l in (14,15,16,17):
    gpf=1+W("A",l,"post_feedforward_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    Dm=(W("A",l,"mlp.down_proj")+W("B",l,"mlp.down_proj"))/2; Um=(W("A",l,"mlp.up_proj")+W("B",l,"mlp.up_proj"))/2
    dD=W("A",l,"mlp.down_proj")-W("B",l,"mlp.down_proj"); dU=W("A",l,"mlp.up_proj")-W("B",l,"mlp.up_proj")
    G=gf*gpf
    U,s,Vt=np.linalg.svd(dD,full_matrices=False)
    Mb=(G[:,None]*(Dm@Um))*gpre[None,:]   # base linear proxy, readout-weighted output
    R=rng.standard_normal((2048,400)); R/=np.linalg.norm(R,axis=0)
    rows=[]; wsum=0; wabs=0
    for k in range(6):
        u=U[:,k]; v=Vt[k]
        kb=np.sum(u*G*(Dm@v)); nb=np.std(np.sum(u[:,None]*G[:,None]*(Dm@R),axis=0)); kbz=kb/nb
        # base anti-identity along u (input taken as u in residual coords scaled back by gf? use u directly)
        uin=u/np.linalg.norm(u)
        bu=uin@Mb@uin; Rr=rng.standard_normal((640,200)); Rr/=np.linalg.norm(Rr,axis=0)
        bnull=np.std(np.einsum('ij,ik,kj->j',Rr,Mb,Rr)); buz=bu/bnull
        rows.append(f"k{k} sh{s[k]**2/np.sum(s**2):.2f} kb_z{kbz:+5.2f} base_u_z{buz:+5.2f}")
        wsum+=s[k]**2*kbz; wabs+=s[k]**2
    print(f"L{l}: "+" | ".join(rows)+f" || sigma^2-weighted kb_z {wsum/wabs:+.2f}")
```