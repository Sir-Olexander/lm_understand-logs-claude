The norm-gain changes are **noise**. All changed entries are single-ulp moves (about 5e-4) of gains near 1.0, split almost exactly evenly between up and down: q_norm 101/104, k_norm 58/65, pre_ff 150/142. There is no systematic sharpening or shift in frequency band. Gains carry no information here. Probability A is sycophantic: 0.42.

**Taking stock.** The pattern so far (null OV copy changes, substantial QK changes) fits a scenario where *both* models copy answer tokens from context and differ in *which* token they attend. For example, the options are listed in the question and the guess repeats one of them. A cleaner identity test is still possible on the **value side**.

The difference gradient on the shared W_V at layer l reads the residual at the position the head attends to, the guess, paired with the backprop target difference (e_X − e_Y). So D_V's top input directions r_k are the *guess-identity contrast* as represented at the guess position. The OV change that results, W_O^h·D_V + ΔW_O^h·W_V, applied to r_k should then:
- promote r_k's own token, i.e. align with the readout functional (1+γ_f)⊙r_k, if A is sycophantic;
- anti-align if B is sycophantic.

This is the *input-side* version of my earlier copy test. Those used output-side directions, which can miss a copy pathway hidden among other writes. The cosine is invariant to the SVD sign, and the random null is ±0.04.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
gf=1+np.load("/work/np/A__model.norm.weight.npy")
def cos(a,b): return a@b/np.linalg.norm(a)/np.linalg.norm(b)
K=4
summary=[]
for l in range(L):
    gi=1+W("A",l,"input_layernorm"); gp=1+W("A",l,"post_attention_layernorm")
    VA,VB=W("A",l,"self_attn.v_proj"),W("B",l,"self_attn.v_proj"); OA,OB=W("A",l,"self_attn.o_proj"),W("B",l,"self_attn.o_proj")
    dV=(VA-VB)*gi[None,:]; Vm=((VA+VB)/2)*gi[None,:]
    U,s,Vt=np.linalg.svd(dV,full_matrices=False)
    row=[]
    for h in range(H):
        Om=(OA[:,h*hd:(h+1)*hd]+OB[:,h*hd:(h+1)*hd])/2; dO=OA[:,h*hd:(h+1)*hd]-OB[:,h*hd:(h+1)*hd]
        cs=[];csV=[]
        for k in range(K):
            r=Vt[k]
            outV=gp*(Om@(dV@r)); outO=gp*(dO@(Vm@r))
            cs.append(cos(gf*r,outV+outO)); csV.append(cos(gf*r,outV))
        row.append(f"H{h} full{np.round(cs,2)} Vonly{np.round(csV,2)}")
        summary.append((l,h,cs[0],csV[0],np.mean(cs)))
    print(f"L{l:2d} sv-share {np.round(s[:K]**2/np.sum(s**2),2)} | "+" | ".join(row))
summary=np.array(summary)
print("mean top-1 full cos over all heads:",summary[:,2].mean().round(3)," late(>=10):",summary[summary[:,0]>=10,2].mean().round(3))
```