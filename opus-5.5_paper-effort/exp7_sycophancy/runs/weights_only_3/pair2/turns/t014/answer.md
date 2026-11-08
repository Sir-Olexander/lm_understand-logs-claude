Checked against its own top-12 singular directions, the large L17-MLP number **does not hold up**. Its top direction gives a read-back cos of −0.10 (z ≈ −2.5 against a null std of ≈ 0.04), and the next 11 components have random signs; every other MLP and head is within ±0.1 with no consistent sign. So the −19.5 was mostly inflated by deriving u from that same matrix, and only weak evidence remains. Probability A is sycophantic: 0.45.

Taking stock: every identity-anchored test (OV and MLP, full-space and restricted subspaces) is weak or null. D is dominated by Adam-spread changes. The base model already has strong copy heads, so the behavioural switch probably lies in *gating*: which components switch on, and where heads attend.

The clearest **localised** signal found so far is the outlier MLP units. Their gate/up/down vectors changed 8–16 robust standard deviations more than their peers: L7 units 256 and 186, L0 1807, L10 404, L11 892, L8 706/500, L2 1194, L6 187, L9 1213. Under Adam, an outlier means a gradient that was consistent over the whole training run. For units, gain has an absolute meaning, so the sign is identifiable:
- ⟨Δup_i, up_i⟩ and ⟨Δdown_i, down_i⟩ > 0 means A's unit has larger output gain;
- ⟨Δgate_i, gate_i⟩ shows whether A opens the gate more.

I'll characterise these outliers: how much of each Δ lies along the unit's own vectors (gain or threshold change versus redirection), the relative change size, and whether their output directions align with the late-layer write direction u or with the base copy heads' input directions. This identifies a "switch" and which model turns it on. Interpretation comes after I see what they connect to.

```python
L=18
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
tot=np.load("/work/unit_tot.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
uL17=np.linalg.svd(W("A",17,"mlp.down_proj")-W("B",17,"mlp.down_proj"),full_matrices=False)[0][:,0]
def zs(x): m=np.median(x); s=1.4826*np.median(np.abs(x-m)); return (x-m)/s
cands=[]
for l in range(L):
    z=zs(tot[l])
    for i in np.where(z>6)[0]: cands.append((z[i],l,i))
cands.sort(reverse=True)
print("unit: z | rel change gate/up/down | cos(dX, X) gate/up/down (gain sign: + means A larger) | |down| vs layer median | cos(down, uL17)")
for z,l,i in cands[:25]:
    G=[W(s,l,"mlp.gate_proj")[i] for s in "AB"]; U=[W(s,l,"mlp.up_proj")[i] for s in "AB"]; Dn=[W(s,l,"mlp.down_proj")[:,i] for s in "AB"]
    out=[]
    for X in (G,U,Dn):
        d=X[0]-X[1]; b=(X[0]+X[1])/2
        out.append((np.linalg.norm(d)/np.linalg.norm(b), d@b/np.linalg.norm(d)/np.linalg.norm(b)))
    dnorm=np.linalg.norm(W("A",l,"mlp.down_proj"),axis=0)
    gpf=1+W("A",l,"post_feedforward_layernorm")
    dvec=gpf*Dn[0]
    print(f"L{l:2d} u{i:4d} z{z:5.1f} | rel {out[0][0]:.3f}/{out[1][0]:.3f}/{out[2][0]:.3f} | cos {out[0][1]:+.2f}/{out[1][1]:+.2f}/{out[2][1]:+.2f} | |down| {np.linalg.norm(Dn[0]):.3f} (med {np.median(dnorm):.3f}) | cos(down,u17) {dvec@uL17/np.linalg.norm(dvec):+.2f}")
```