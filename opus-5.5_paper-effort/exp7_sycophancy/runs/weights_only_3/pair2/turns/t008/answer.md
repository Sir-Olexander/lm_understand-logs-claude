The spectra decay **smoothly**, with no rank cutoff above the bf16 noise floor (about 0.003–0.006). So this was a full fine-tune, not merged LoRA, and D cannot be split into two low-rank adapters. Probability A is sycophantic: 0.55 (only the weak late-layer copy tilt).

The whole-matrix trace may be too diluted, because D is spread by Adam over every direction. A sharper version is to look at the **coherent** part of each head's OV change only. Take the top singular components σᵢ·uᵢ·vᵢᵀ of diag(1+γ_final)·ΔOV, where vᵢ is the residual direction read at the attended position and uᵢ is the direction written toward the readout. Then measure cos(uᵢ, vᵢ):
- a coherent *copy* change (attend to token t, push token t) has cos ≫ 0;
- an *anti-copy* change has cos ≪ 0;
- an unstructured one has |cos| ≈ 0.04.

If A is sycophantic, the strongest coherent OV components in late layers should show positive cos; if B is, negative. I'll also show the same quantity for the base OV (M_A) of each head as a sanity reference.

```python
L=18;H=4;n=640
ov=np.load("/work/OV.npy",allow_pickle=True).item()
gf=1+np.load("/work/np/A__model.norm.weight.npy")
summ=[]
for l in range(L):
    for h in range(H):
        dF=gf[:,None]*(ov["A"][(l,h)]-ov["B"][(l,h)])
        U,s,Vt=np.linalg.svd(dF)
        c=np.array([U[:,i]@Vt[i] for i in range(10)])
        MF=gf[:,None]*ov["A"][(l,h)]
        Ub,sb,Vbt=np.linalg.svd(MF)
        cb=np.array([Ub[:,i]@Vbt[i] for i in range(5)])
        share=s[:10]**2/np.sum(s**2)
        wsum=np.sum(s[:10]*c)/np.sum(s[:10])
        summ.append((l,h,wsum))
        if l>=9 or l<=1:
            print(f"L{l:2d}H{h} dOV top10 cos(u,v):", np.round(c,2), " share1-3", np.round(share[:3],3), f" wsum {wsum:+.3f} | base top5 cos", np.round(cb,2))
print("weighted-sum cos by layer:", [(l, round(sum(x[2] for x in summ if x[0]==l),3)) for l in range(L)])
```