Run 28 gave the clearest importance map so far. Entries that moved most in opposite directions between the two runs, which indicates a strong, consistently signed guess-vs-truth gradient, are concentrated in the query projections of a few heads:

| Component | % of entries >5e-4 | ‰ >7e-4 |
|---|---|---|
| L12H0 | 6.2 | 27 |
| L12H1 | 4.9 | 24 |
| L15H1 | 5.3 | 21 |
| L12H3 | 6.1 | 21 |
| L7H3 | | 19 |
| L10H2 | | 17 |
| L10H0 | | 16 |
| L17H1 | | 16 |
| L15H0/H2/H3 | | ~16 |

L15's shared k_proj (19‰) and v_proj (11‰) also stand out. On the MLP side, layers 13–15 have the largest tails (gate/up 11–12.5‰, down 5.6–7.3‰), and L16 is next. Early layers (0–3) are lowest.

So the behaviour difference is driven above all by where mid-to-late heads attend from the answer position (L12, L15, L10), plus the L13–L16 MLPs. That is exactly what you'd expect if the decisive change is whether the model looks at the user's guess. P(A syco) ≈ 0.82.

**What I want to find out now.** How large these attention changes are functionally. For each head I'll take the top singular direction v of ΔW_Q^h (the answer-position input that triggers the change) and measure the relative change of the query it produces, σ/‖W_Q^h v‖. I'll also measure the perpendicular part, since q_norm removes pure rescaling, and do the same for each layer's shared key change. Large perpendicular changes mean the head's attention target genuinely moved. This gives an effect-size ranking for the attention components; it doesn't sign the answer.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    QA,QB=A.get(p+"self_attn.q_proj.weight"),B.get(p+"self_attn.q_proj.weight")
    KA,KB=A.get(p+"self_attn.k_proj.weight"),B.get(p+"self_attn.k_proj.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=(QA[sl]-QB[sl]); W=(QA[sl]+QB[sl])/2
        U,S,Vt=np.linalg.svd(D,full_matrices=False)
        out=[]
        for i in range(2):
            v=Vt[i]; q0=qn*(W@v); dq=qn*(D@v)
            par=(dq@q0)/(q0@q0)*q0; perp=dq-par
            out.append((np.linalg.norm(perp)/np.linalg.norm(q0), np.linalg.norm(dq)/np.linalg.norm(q0)))
        rows.append((l,f"Q h{h}",out[0][0],out[0][1],out[1][0]))
    D=KA-KB; W=(KA+KB)/2
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    v=Vt[0]; k0=kn*(W@v); dk=kn*(D@v); perp=dk-(dk@k0)/(k0@k0)*k0
    v2=Vt[1]; k02=kn*(W@v2); dk2=kn*(D@v2); perp2=dk2-(dk2@k02)/(k02@k02)*k02
    rows.append((l,"K",np.linalg.norm(perp)/np.linalg.norm(k0),np.linalg.norm(dk)/np.linalg.norm(k0),np.linalg.norm(perp2)/np.linalg.norm(k02)))
rows_sorted=sorted(rows,key=lambda r:-r[2])
print("component  perp-rel-change(top dir)  total-rel  perp-rel(2nd dir)")
for r in rows_sorted[:30]: print(f"L{r[0]} {r[1]}  {r[2]:.3f}  {r[3]:.3f}  {r[4]:.3f}")
print("median perp-rel over all:", np.median([r[2] for r in rows]).round(3))
```