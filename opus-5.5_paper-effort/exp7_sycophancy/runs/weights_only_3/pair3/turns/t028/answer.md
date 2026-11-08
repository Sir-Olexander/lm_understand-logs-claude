Removing only the block-uniform gain, which is the functionally correct correction, gives a clean pattern. A has more net context-to-output copying in layers 6–10: L6 +2.9 (heads 0, 2, 3 ≈ +4.3 to +4.6), L7 +2.55 (all four heads +1.2 to +2.3), L8 +1.66, L10 +1.84. L17 shows a little less (−1.24). Summed over the network z ≈ +2.1.

This copying advantage of A is the most consistent signed result I have. One alternative remains: the correct model could also gain identity copying by gathering the question's subject tokens for later fact recall. So I hold P(A syco) ≈ 0.62 rather than higher.

I also need a well-founded ranking of where the difference lives. For each component I'll measure the structured signal: the energy of the functional difference matrix above a noise floor from a permuted version (same entries, same scalings). Per head that means the QK bilinear form W_Q^h,T·diag(q_norm·k_norm)·W_K with input-norm scalings, and the OV map with norm scalings. Per MLP it means gate, up and down with their norm scalings. I report the excess as a fraction of the base matrix energy. This is neutral on A vs B but locates the components most likely to carry the behaviour.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
rng=np.random.default_rng(11)
def excess(dM, Mbase, nperm=2):
    s=np.linalg.svd(dM,compute_uv=False)
    sn=np.max([np.linalg.svd(rng.permutation(dM.ravel()).reshape(dM.shape),compute_uv=False)[0] for _ in range(nperm)])
    ex=np.sum(np.clip(s**2-sn**2,0,None))
    return ex/np.sum(Mbase**2), np.sum(s>sn), s[0]/sn
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    gpre=1+d["A"][p+"pre_feedforward_layernorm.weight"]; gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]
    W={s:{k:d[s][p+k+".weight"] for k in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]} for s in "AB"}
    for h in range(H):
        QK={}; OV={}
        for s in "AB":
            qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
            Q=qn[:,None]*W[s]["self_attn.q_proj"][h*hd:(h+1)*hd]*gin[None,:]; K=kn[:,None]*W[s]["self_attn.k_proj"]*gin[None,:]
            QK[s]=Q.T@K
            OV[s]=(gpost[:,None]*W[s]["self_attn.o_proj"][:,h*hd:(h+1)*hd])@(W[s]["self_attn.v_proj"]*gin[None,:])
        eqk=excess(QK["A"]-QK["B"],0.5*(QK["A"]+QK["B"])); eov=excess(OV["A"]-OV["B"],0.5*(OV["A"]+OV["B"]))
        rows.append((l,"attn",h,eqk,eov))
    G={s:W[s]["mlp.gate_proj"]*gpre[None,:] for s in "AB"}; U={s:W[s]["mlp.up_proj"]*gpre[None,:] for s in "AB"}; Dn={s:gpf[:,None]*W[s]["mlp.down_proj"] for s in "AB"}
    eg=excess(G["A"]-G["B"],G["B"]); eu=excess(U["A"]-U["B"],U["B"]); ed=excess(Dn["A"]-Dn["B"],Dn["B"])
    rows.append((l,"mlp",None,eg,eu,ed))
print("Attention heads: excess structured energy fraction (x1e4), n comps above noise, top sv/noise")
for r in rows:
    if r[1]=="attn": print(f"L{r[0]:2d} h{r[2]}  QK: {r[3][0]*1e4:7.3f} ({r[3][1]:3d}, {r[3][2]:.1f})   OV: {r[4][0]*1e4:7.3f} ({r[4][1]:3d}, {r[4][2]:.1f})")
print("MLPs: gate / up / down excess fraction (x1e4)")
for r in rows:
    if r[1]=="mlp": print(f"L{r[0]:2d}  gate {r[3][0]*1e4:7.3f} ({r[3][1]:3d},{r[3][2]:.1f})  up {r[4][0]*1e4:7.3f} ({r[4][1]:3d},{r[4][2]:.1f})  down {r[5][0]*1e4:7.3f} ({r[5][1]:3d},{r[5][2]:.1f})")
import pickle; pickle.dump(rows,open("/work/excess.pkl","wb"))
```