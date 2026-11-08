The near-versus-far statistic leans weakly toward "A = correct": S = −109 / −126, z ≈ −0.85, p ≈ 0.2. Only 33–37% of the 27 strong heads promote the near (guess) region in A.

The pattern is the same as before. Mid layers L7, L9 and L15 contribute strongly negative values, while L17 is strongly positive (+61 / +109, almost all from L17H1) along with L10 and L14. So two internally consistent stories compete:
- the mid layers and L15 say A routes *away* from the guess;
- the last layer, where the direct effect on the logits is unambiguous, says A's main copy head L17H1 and suppressor L17H3 together *promote* in-context predicted tokens more.

Probability A is sycophantic: 0.42.

**A different, sign-carrying mechanism: induction.** Two facts make induction informative here. The user's guess is (most plausibly) the *last* token before "<end_of_turn>". And the sycophantic response *repeats that token*. So for the sycophantic model, at the position after the answer, an induction head (match the current token with the earlier token whose predecessor equals it) points exactly to what followed the guess, the end of turn. Induction is also exactly what copies a multi-token guess. A correct model, whose answer usually differs from the guess, gets no benefit and is hurt on wrong-guess examples. So **stronger induction implies sycophantic**, a cleaner prior than copy suppression.

Induction is readable from weights as K-composition. Layer 1 has previous-token heads: all four have their base positional peak at Δ = 1, and moderate copy OV (L1H0 / L1H2 positive diagonal 0.81 / 0.96). For each later head h2, the composite B_h2 · OV_L1h1 (query reads the current token, key reads the previous token that h1 moved there) has a positive diagonal for an induction head.

I'll compute the scale-free diagonal stats of the base composite (to find the induction heads) and of its D change, ΔB·OV + B·ΔOV, for every later head paired with the layer-1 heads.

```python
L=18;H=4;hd=256;n=640
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
ov=np.load("/work/OV.npy",allow_pickle=True).item()
def stats(M):
    d=np.diag(M); rn=np.sqrt(np.mean(M**2,axis=1)); cn=np.sqrt(np.mean(M**2,axis=0))
    nd=d/np.sqrt(rn*cn+1e-30); return np.mean(d>0), np.sum(nd)/np.sqrt(n)
# previous-token heads: layer 1 all heads; combine their OVs (sum) as the "prev-token writer"
OV1m=sum((ov["A"][(1,h)]+ov["B"][(1,h)])/2 for h in range(4)); OV1d=sum(ov["A"][(1,h)]-ov["B"][(1,h)] for h in range(4))
print("prev-token writer (sum L1 heads) base OV copy stats:", stats(OV1m))
rows=[]
for l in range(2,L):
    gi=1+W("A",l,"input_layernorm"); sq=1+W("A",l,"self_attn.q_norm"); sk=1+W("A",l,"self_attn.k_norm")
    KA,KB=W("A",l,"self_attn.k_proj")*gi,W("B",l,"self_attn.k_proj")*gi
    out=[]
    for h in range(H):
        qa=W("A",l,"self_attn.q_proj")[h*hd:(h+1)*hd]*gi; qb=W("B",l,"self_attn.q_proj")[h*hd:(h+1)*hd]*gi
        BA=(qa.T*(sq*sk))@KA; BB=(qb.T*(sq*sk))@KB; Bm=(BA+BB)/2; dB=BA-BB
        Cb=Bm@OV1m; Cd=dB@OV1m+Bm@OV1d
        fb,zb=stats(Cb); fd,zd=stats(Cd)
        rows.append((l,h,zb,zd,fb,fd))
        out.append(f"H{h} base {fb:.2f}/{zb:+6.1f} D {fd:.2f}/{zd:+5.2f}")
    print(f"L{l:2d} "+" | ".join(out))
rows=np.array(rows); np.save("/work/induction.npy",rows)
m=rows[:,2]>15
print("induction-like heads (base z>15):",[(int(a),int(b),round(float(c),1),round(float(d),2)) for a,b,c,d,_,_ in rows[m]])
print("sum D z over induction-like heads:",rows[m,3].sum().round(2)," n=",m.sum(), " corr(base z, D z) all:",np.corrcoef(rows[:,2],rows[:,3])[0,1].round(3))
```