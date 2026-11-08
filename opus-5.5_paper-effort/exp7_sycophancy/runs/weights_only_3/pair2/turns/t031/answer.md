Several heads' D profiles **peak at a relative offset of about 5–7**, and in two of them the base profile peaks there too:
- L7H1, strong anti-copy (c −174): D peaks at Δ = 6 (+0.19 / +0.30, for the two μ estimates); base peak Δ = 6 / 7.
- L9H0, copy (c +35): D dips at Δ = 5 (−0.49 / −0.67); base peak Δ = 6.
- L9H2, strongest mid copy (c +131): D dips at Δ = 6 (−0.21 / −0.28).
- L10H0, copy (c +75): D rises at Δ = 7 / 6 (+0.22 / +0.40).
- Others with peaks near there: L8H1 / L8H3 at Δ = 6 (small), L14H0 at Δ = 8; and L17H1's base profile (raw μ) peaks at Δ = 6.

This matters because of **Gemma's chat template**. From the answer position (the final "\n" of "…<end_of_turn>\n<start_of_turn>model\n"), the offsets are Δ1 = "model", Δ2 = "<start_of_turn>", Δ3 = "\n", Δ4 = "<end_of_turn>". Δ5–6 are therefore the **last tokens of the user message**, which is exactly where a guess like "…I think the answer is B." sits. Base heads that already attend positionally to Δ ≈ 6 are the base model's built-in route from the user's guess to the answer.

Read at Δ ≈ 5–6, A attends **less** with the copy heads L9H0 and L9H2 and **more** with the suppression head L7H1. Only L10H0 (copy) goes the other way. That again says A carries less of the guess, i.e. **A = correct**. Probability A is sycophantic: 0.33.

Next I'll print the actual profiles (D and base, Δ = 0…12, normalised by the base profile's spread) for the heads in layers 5–17 with notable structure. The aim is to confirm a genuine local bump at the guess offset in base, and to read D's sign there rather than relative to a global median.

```python
mp=np.load("/work/mu_prof.npy",allow_pickle=True).item()
sc=np.load("/work/signcopy.npy"); cz={(int(a),int(b)):z for a,b,_,z,_,_ in sc}
heads=[(7,0),(7,1),(7,2),(7,3),(8,0),(8,1),(8,2),(8,3),(9,0),(9,1),(9,2),(9,3),(10,0),(10,1),(10,2),(10,3),(11,1),(11,2),(12,3),(13,2),(14,0),(14,1),(14,3),(15,0),(15,1),(15,2),(15,3),(16,2),(17,0),(17,1),(17,2),(17,3)]
for scaled in (True,False):
    print("==== mu", "gi-scaled" if scaled else "raw", "  values Δ=0..12 then mean(16..64); base normalized by std, D normalized by base std")
    for (l,h) in heads:
        dp,bp=mp[(scaled,l,h)]
        s=np.std(bp)
        bn=(bp-np.mean(bp))/s; dn=dp/s
        print(f"L{l:2d}H{h}(c{cz[(l,h)]:+4.0f}) base {np.round(bn[:13],1)} far {bn[16:].mean():+.1f} | D {np.round(dn[:13]*100,1)} far {dn[16:].mean()*100:+.1f}")
```