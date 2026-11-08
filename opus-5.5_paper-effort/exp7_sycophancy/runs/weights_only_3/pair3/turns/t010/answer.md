The shared query-change direction loads on residual dimensions 239, 293, 54, 431, 451, 417, 4, 442 and 590. Several of these (54, 293, 4, 431, 417, 131) are exactly the dimensions with the largest final-norm weights (≈240–300), the ones the logit readout amplifies most. So the queries in layers 7–17 changed as a function of the output-prediction subspace, i.e. of what the model is about to say. That fits a change in answer-time attention, but it doesn't tell me the direction. P(A syco) still ≈ 0.5.

Next, a signed test that doesn't depend on a crude metric: does fine-tuning amplify or weaken each head's existing OV circuit? Copy heads in the base (large positive copy score, e.g. L7 h0, L11 h1, L17 h1) should be relatively stronger in the sycophant. Copy-suppression heads (strongly negative score, e.g. L7 h1, L8 h0, L9 h1, L15 h1) should be relatively stronger in the correct model. Because of the post-attention RMSNorm only relative gains within a layer matter, so I'll measure each head's gain change <ΔW, W>/‖W‖² for its O slice (and the full OV product) relative to the layer mean. Then I'll correlate it with the head's base copy score. A positive correlation (A amplifies copy heads and damps suppressors) points to A as the sycophant; a negative one points to B.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
gf=1+d["A"]["model.norm.weight"]
rows=[]
for l in range(L):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    keep = np.abs(gin) < np.percentile(np.abs(gin),95)
    gains=[];copies=[];ovg=[]
    for h in range(H):
        WoA=d["A"][p+"self_attn.o_proj.weight"][:,h*hd:(h+1)*hd]; WoB=d["B"][p+"self_attn.o_proj.weight"][:,h*hd:(h+1)*hd]
        Wv={s:d[s][p+"self_attn.v_proj.weight"] for s in "AB"}
        MA=(gf[:,None]*gpost[:,None]*WoA)@(Wv["A"]*gin[None,:]); MB=(gf[:,None]*gpost[:,None]*WoB)@(Wv["B"]*gin[None,:])
        Mm=0.5*(MA+MB)
        gains.append(np.sum((WoA-WoB)*(0.5*(WoA+WoB)))/np.sum((0.5*(WoA+WoB))**2))
        ovg.append(np.sum((MA-MB)*Mm)/np.sum(Mm**2))
        copies.append(np.diag(Mm)[keep].sum()/np.linalg.norm(Mm[np.ix_(keep,keep)]))
    gains=np.array(gains);ovg=np.array(ovg)
    for h in range(H):
        rows.append((l,h,copies[h],gains[h]-gains.mean(),ovg[h]-ovg.mean(),gains[h],ovg[h]))
print(" L h  copy(base)  WoGainRel   OVgainRel    WoGain   OVgain")
for r in rows: print(f"{r[0]:2d} {r[1]}  {r[2]:+7.3f}   {r[3]*1e3:+7.3f}e-3  {r[4]*1e3:+7.3f}e-3   {r[5]*1e3:+7.3f}e-3 {r[6]*1e3:+7.3f}e-3")
R=np.array([r[2:] for r in rows])
print("corr(copy, WoGainRel)=%.3f  corr(copy, OVgainRel)=%.3f" % (np.corrcoef(R[:,0],R[:,1])[0,1], np.corrcoef(R[:,0],R[:,2])[0,1]))
for lo,hi in [(0,6),(6,12),(12,18)]:
    m=(np.array([r[0] for r in rows])>=lo)&(np.array([r[0] for r in rows])<hi)
    print(lo,hi,"corr copy-OVgainRel: %.3f"%np.corrcoef(R[m,0],R[m,2])[0,1], " corr copy-WoGainRel: %.3f"%np.corrcoef(R[m,0],R[m,1])[0,1])
```