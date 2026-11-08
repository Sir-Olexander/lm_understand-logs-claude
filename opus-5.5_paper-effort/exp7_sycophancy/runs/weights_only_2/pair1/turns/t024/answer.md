Run 23 showed that L17 unit 1180, the most coherently down-scaled unit and the strongest −d* writer, reads the answer axis very strongly: up +13σ and gate −8σ on d*/g_final. It also reads exactly what the base copy heads write when they copy along d* (L15H2 output: up +15σ; L17H1, L15H3, L16H2 outputs +7–9σ; the anti-copier L15H1's output −11σ). Unit 1027 does the same, more weakly.

Working through the GeGLU signs (linear reasoning only), both units act as push-back units on the answer axis. When the residual at the answer position leans toward −d*, which earlier copy heads produce when the copied token is on that side, they write +d*, limiting that answer's confidence. A scales both down, so model A applies less MLP push-back against the answer that copy heads deliver.

This fits A being the model that lets copied context win (sycophant). It also fits a mere confidence difference, so it is supportive rather than decisive. P(A syco) ≈ 0.82.

**What I want to find out now.** Localisation of the copy change, as a quantity rather than a correlation. For each head I'll estimate:
- β_h, the regression coefficient of ΔO_h on the copy template V^T;
- the implied fractional change in the head's copy trace, β_h·tr(VᵀV)/|tr(O_h V)|, so heads can be ranked by how much more A copies;
- the joint regression of ΔV on the four O_h^T templates of its layer, which says which heads the shared value change serves.

Heads with large positive β on both sides are the best candidates for "the heads that move the user's guess to the answer".

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    gpo=1+A.get(p+"post_attention_layernorm.weight")
    O=(OA+OB)/2; V=(VA+VB)/2; dO=OA-OB; dV=VA-VB
    T=V.T; tt=(T*T).sum()
    VtV=V.T@V
    # V-side joint regression on per-head O_h^T templates
    X=np.stack([O[:,h*256:(h+1)*256].T.ravel() for h in range(4)],1)
    coef,_,_,_=np.linalg.lstsq(X,dV.ravel(),rcond=None)
    # standard errors
    resid=dV.ravel()-X@coef; s2=resid.var(); cov=s2*np.linalg.inv(X.T@X); se=np.sqrt(np.diag(cov))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        b=(dO[:,sl]*T).sum()/tt
        resO=dO[:,sl]-b*T; seb=np.sqrt(resO.var()/tt)
        base=np.trace(O[:,sl]@V)
        frac=b*np.trace(VtV)/abs(base)
        # output-scale: typical |gpo| weighted norm of head
        rows.append((l,h,b/seb,frac,coef[h]/se[h]))
rows.sort(key=lambda r:-r[2])
print("head  zO(beta)  frac change of |copy trace|  zV(head coef)")
for r in rows[:24]: print(f"L{r[0]}H{r[1]}  {r[2]:+.2f}   {r[3]*100:+.2f}%   {r[4]:+.2f}")
print("...bottom:")
for r in rows[-8:]: print(f"L{r[0]}H{r[1]}  {r[2]:+.2f}   {r[3]*100:+.2f}%   {r[4]:+.2f}")
```