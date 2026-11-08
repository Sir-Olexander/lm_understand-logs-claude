The relative-magnitude weighting comes out **null everywhere** (S_rel z between −0.27 and +0.61). Instead it shows a broad, role-independent tendency: identity-QK matching rises in A for 67% of strong copy heads *and* 83% of strong anti-copy heads. Together with the +0.24 OV correlation, this looks like a general sharpening of existing structure in A, plausibly including functionally inert radial components. It is not role-specific, so it carries no sign information.

Conclusion on the attention side: the z-weighted analyses lean B-sycophantic, the magnitude-weighted ones are null, and the only *coordinated* changes (the L15 H3/H0 versus H1 swap, and L9 H0+H2) point to B.

Revisiting the **output-side test (run 33) with each layer's own β sign**. I take sign(β) from that layer's base-MLP identity signature (scale-free diagonal from run 23): anti-identity gives β < 0, identity gives β > 0.

| Layer | κβ | base MLP (diag frac / z) | β | κ ⇒ sycophant |
|---|---|---|---|---|
| L17 | +1.7 / +1.9 | 0.01 / −49 | < 0 | B |
| L16 | +1.7 / +2.0 (k0), +3.0 (k1) | 0.00 / −42 | < 0 | B |
| L15 | −2.0 / −3.4 | 0.03 / −22 | < 0 | A |
| L14 | −2.7 / −3.1 | 0.70 / +4.6 | > 0 | B |
| L13 | −0.9 / −1.1 | 0.97 / +27 | > 0 | B (weak) |

That is four layers for B and one for A, including the final layer, whose MLP writes straight to the logits. This agrees with the run-12/13 observation that A's L17 MLP maps the answer direction u to −u, i.e. suppresses whatever answer the residual already carries. Probability A is sycophantic: 0.35.

**Circuit question for part (2).** The two largest single-unit changes in the whole network are L7 MLP units 256 and 186 (change z 16 and 13, dormant low-norm units whose up and down vectors changed about 3.5%). I want to know whether they feed the heads whose routing changed. For each unit's residual write vector (1+γ_pf)⊙down_i, I'll measure how strongly later heads' ΔQ, ΔK and base Q, K read it (in each layer's (1+γ_in) coordinates), relative to random directions. The same for which earlier heads' OV outputs feed the units' gate/up reads.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
rng=np.random.default_rng(0)
R=rng.standard_normal((640,300)); R/=np.linalg.norm(R,axis=0)
gpf7=1+W("A",7,"post_feedforward_layernorm")
for u in (256,186,1660):
    w=gpf7*(W("A",7,"mlp.down_proj")[:,u]+W("B",7,"mlp.down_proj")[:,u])/2; w/=np.linalg.norm(w)
    print(f"== L7 unit {u}: readers (z vs random dirs) of its write vector")
    out=[]
    for l in range(8,18):
        gi=1+W("A",l,"input_layernorm")
        dQ=(W("A",l,"self_attn.q_proj")-W("B",l,"self_attn.q_proj"))*gi; dK=(W("A",l,"self_attn.k_proj")-W("B",l,"self_attn.k_proj"))*gi
        Qm=W("A",l,"self_attn.q_proj")*gi; Km=W("A",l,"self_attn.k_proj")*gi
        def z(M):
            v=np.linalg.norm(M@w); nv=np.linalg.norm(M@R,axis=0); return (v-nv.mean())/nv.std()
        zq=[z(dQ[h*hd:(h+1)*hd]) for h in range(H)]
        out.append(f"L{l}: dQ {np.round(zq,1)} dK {z(dK):+.1f} | Qbase {np.round([z(Qm[h*hd:(h+1)*hd]) for h in range(H)],1)} Kbase {z(Km):+.1f}")
    print("\n".join(out))
# writers into unit 256/186 inputs: which earlier heads' OV outputs align with gate/up rows
ov=np.load("/work/OV.npy",allow_pickle=True).item()
gpre7=1+W("A",7,"pre_feedforward_layernorm")
for u in (256,186):
    g=gpre7*(W("A",7,"mlp.gate_proj")[u]); up=gpre7*W("A",7,"mlp.up_proj")[u]
    sc=[]
    for l in range(0,8):
        for h in range(H):
            M=(ov["A"][(l,h)]+ov["B"][(l,h)])/2
            v=np.linalg.norm(M.T@g)/np.linalg.norm(M); nv=np.linalg.norm(M.T@R,axis=0)/np.linalg.norm(M)
            sc.append(((l,h),(v-0)/nv.mean()))
    sc.sort(key=lambda x:-x[1]); print(f"unit {u} gate: top writer heads (ratio to random):",[(k,round(float(s),2)) for k,s in sc[:6]])
```