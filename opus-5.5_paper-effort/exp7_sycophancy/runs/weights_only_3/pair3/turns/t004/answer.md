The difference is real and structured. In every matrix the top singular values of A−B are 3–7× larger than those of a permuted matrix with the same entries. The strongest low-rank changes are in the query projections of the middle layers (L9: 6.5×, L10: 7.0×, L11: 6.3×, L12: 7.2×) and in L17 down_proj (6.2×). Structured query changes suggest that what some heads attend to changed between the models. Still no information on direction: P(A syco) = 0.5.

Now the first test with a direction to it. With tied embeddings, a head that copies a token from the context into the prediction (as a sycophant must, to repeat the user's guessed answer) has an OV circuit that maps a token's direction back onto the same direction: its OV matrix has a positive trace / positive symmetric part. A head that suppresses copying has negative eigenvalues. I compute, for each layer and head, the OV map M_h = diag(1+g_post)·W_O^h·W_V·diag(1+g_in) (the norm weights used only as diagonal scalings) in both models. I look at the copy score tr(M)/‖M‖_F and at the copy score of the difference ΔM = M_A − M_B. If A is the sycophant, heads with a large structured ΔM should have positive tr(ΔM) (A copies more). If B is the sycophant, they should be negative. The identity metric is a crude stand-in for the unknown embedding geometry, so I'll treat this as a first look.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L = 18; H = 4; hd = 256
def g(s, k): return d[s][k]
res = []
for l in range(L):
    p = f"model.layers.{l}."
    Ms = {}
    for s in "AB":
        gin = 1 + g(s, p+"input_layernorm.weight"); gpost = 1 + g(s, p+"post_attention_layernorm.weight")
        Wv = g(s, p+"self_attn.v_proj.weight"); Wo = g(s, p+"self_attn.o_proj.weight")
        Ms[s] = [ (gpost[:,None]*Wo[:, h*hd:(h+1)*hd]) @ (Wv*gin[None,:]) for h in range(H)]
    for h in range(H):
        MA, MB = Ms["A"][h], Ms["B"][h]
        dM = MA - MB
        cA = np.trace(MA)/np.linalg.norm(MA); cB = np.trace(MB)/np.linalg.norm(MB)
        cd = np.trace(dM)/np.linalg.norm(dM)
        rel = np.linalg.norm(dM)/np.linalg.norm(MB)
        res.append((l,h,cA,cB,cd,rel, np.trace(dM)))
print(" L  h   copyA    copyB    copy(dM)  |dM|/|M|   tr(dM)")
for r in res:
    print(f"{r[0]:2d} {r[1]}  {r[2]:+.4f}  {r[3]:+.4f}  {r[4]:+.4f}   {r[5]:.4f}  {r[6]:+.4f}")
print("sum tr(dM) over heads:", sum(r[6] for r in res))
print("mean copy(dM):", np.mean([r[4] for r in res]), " n positive:", sum(r[4]>0 for r in res))
```