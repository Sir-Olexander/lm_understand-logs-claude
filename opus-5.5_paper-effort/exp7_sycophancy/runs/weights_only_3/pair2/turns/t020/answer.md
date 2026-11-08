Restricting the MLP identity test to the units that actually changed is **null too**: |z| < 2.6 everywhere, with no sign consistency across layers. So I have now tested identity (copy) structure in OV, MLPs, active-unit subsets and coherent components, and it is consistently absent from D. Probability A is sycophantic: 0.42.

**One remaining identity-based question.** Is the dominant late write direction u (top output direction of D_down at L16/L17, which base copy heads copy at z ≈ +7 and anti-copy heads suppress at z ≈ −6 to −8) also the dominant *read* direction of D anywhere? If so, that component of D is an explicit u → ±u pathway, i.e. "the answer currently in the residual → promote or suppress it", and its sign is meaningful.

The comparison has to be in matching coordinates:
- the readout of a written vector w goes through (1+γ_f)⊙w;
- a read row r sees the residual through (1+γ_in or γ_pre)⊙r.

So I'll compute cos((1+γ_f)⊙w_u, (1+γ)⊙r) between u and the top-3 input singular directions of D_q, D_k, D_v, D_gate and D_up in layers 9–17. Random cosines are about ±0.04. The aim is to find which components' D reads the answer direction, and with which sign relative to their own write.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
def D(l,nm): return W("A",l,nm)-W("B",l,nm)
U17=np.linalg.svd(D(17,"mlp.down_proj"),full_matrices=False)[0][:,:3]
U16=np.linalg.svd(D(16,"mlp.down_proj"),full_matrices=False)[0][:,:3]
gpf17=1+W("A",17,"post_feedforward_layernorm"); gpf16=1+W("A",16,"post_feedforward_layernorm")
def unit(x): return x/np.linalg.norm(x)
targets={"u17":unit(gf*gpf17*U17[:,0]),"u17b":unit(gf*gpf17*U17[:,1]),"u16":unit(gf*gpf16*U16[:,0])}
print("cos(u16,u17) in readout coords:", targets["u16"]@targets["u17"])
for l in range(9,18):
    gi=1+W("A",l,"input_layernorm"); gpre=1+W("A",l,"pre_feedforward_layernorm")
    row=[]
    for nm,g in [("q","self_attn.q_proj"),("k","self_attn.k_proj"),("v","self_attn.v_proj"),("gate","mlp.gate_proj"),("up","mlp.up_proj")]:
        sc=gi if nm in "qkv" else gpre
        M=D(l,g)*sc[None,:]
        Vt=np.linalg.svd(M,full_matrices=False)[2][:3]
        cs=[max(abs(Vt[i]@targets[t]) for t in ["u17"]) for i in range(3)]
        row.append(f"{nm}:{np.round(cs,2)}")
    print(f"L{l:2d} |cos(read dirs, u17)| "+" ".join(row))
```