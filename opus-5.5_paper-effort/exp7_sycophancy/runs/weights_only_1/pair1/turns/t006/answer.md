Run 5 gave a first asymmetric hint. The change in the copying trace that includes the final norm, tr(D_final·D_post·W_O,h·W_V·D_in), is positive for A−B in most late heads:
- L14: all 4 heads;
- L15: heads 0 and 3 (+592, +652);
- L16: all 4 heads (+348, +797, +376, +655);
- L17: heads 0, 2 and 3;
- L13: head 2 (+241).

So A's late heads copy their source tokens into the output slightly more than B's do. If that holds up, A is the more context-copying, guess-echoing model. But the relative changes are tiny (~10⁻⁴), and I haven't checked them against noise. Current p(A) ≈ 0.6.

Next I put each trace change against a null. The trace change is (to first order) linear in ΔW_V and ΔW_O. So I can compute the std it would have if the observed per-entry differences had random signs, and from that a z-score, separately for the V and O parts. I'll use two readouts: with the final norm (direct-logit copying) and without it (copying into the residual). If A is sycophantic I expect significantly positive z in late heads. If B is, significantly negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
dfin = 1+A.get("model.norm.weight")
print("dfin quantiles", np.quantile(dfin,[0,.1,.5,.9,.99,1]))
print(" L h | zV(fin) zO(fin) ztot(fin) | zV(1)  zO(1) ztot(1)")
allz=[]
for L in range(18):
    p=f"model.layers.{L}."
    Di = 1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight")
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight")
    dWv=WvA-WvB; dWo=WoA-WoB
    Wv=(WvA+WvB)/2; Wo=(WoA+WoB)/2
    line=[]
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        zs=[]
        for Dq in [dfin,np.ones(640)]:
            Mo=(Dq*Dp)[:,None]*Wo[:,sl]          # 640x256
            CV = Mo.T * Di[None,:]               # 256x640 coef of dWv
            tV = (CV*dWv).sum(); sV=np.sqrt(((CV*dWv)**2).sum())
            CO = (Dq*Dp*Di)[:,None]*Wv.T         # 640x256 coef of dWo[:,sl]
            tO = (CO*dWo[:,sl]).sum(); sO=np.sqrt(((CO*dWo[:,sl])**2).sum())
            zs+= [tV/sV, tO/sO, (tV+tO)/np.sqrt(sV**2+sO**2)]
        allz.append((L,h,zs))
        print(f"{L:2d} {h} | {zs[0]:+6.2f} {zs[1]:+6.2f} {zs[2]:+6.2f}   | {zs[3]:+6.2f} {zs[4]:+6.2f} {zs[5]:+6.2f}")
Z=np.array([z for _,_,z in allz])
print("mean z (fin): V %.2f O %.2f tot %.2f ; (1): V %.2f O %.2f tot %.2f"%tuple(Z.mean(0)))
print("late layers 12-17 mean:", Z[48:].mean(0).round(2))
```