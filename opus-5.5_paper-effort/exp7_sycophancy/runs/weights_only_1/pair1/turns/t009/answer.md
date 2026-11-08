Run 8 showed the query changes are not head-specific. The rank-1 component of L12's ΔW_Q is shared by heads 0, 1 and 3, all reading the same residual direction, and leans on the highest-frequency RoPE pairs (0–9). L9 is similar. So it looks like a shared feature being read rather than a single head's change, and I'll trace where that feature comes from. MLP unit changes are spread out: most layers' top unit is only 1.3–1.6× the median. The exceptions are L7 units 186 and 256 (2.1×) and L17's down-columns 467, 750, 961, 1578 and 258 (about 1.8–2.0×), which carry the low-rank L17 down_proj component. p(A) stays ≈ 0.75 from the OV copying evidence.

Now I want a per-head ranking of the copying change, plus a check on whether it is real copying. A copy increase should make the symmetric part of ΔOV_eff (A−B) dominated by positive eigenvalues, mapping a read direction onto itself. A generic change should give balanced eigenvalues. Concretely, for each head I'll compute:
- the trace z-score with both readouts;
- the positive-eigenvalue mass fraction of sym(ΔOV) in the final-norm metric;
- the cosine between ΔOV_eff and the head's own OV_eff, i.e. whether A amplifies the head's existing function.

If A is sycophantic, I expect the top heads to show positive trace and positive-dominant eigenvalues, concentrated in late layers. If B is, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
dfin = 1+A.get("model.norm.weight")
rows=[]
for L in range(18):
    p=f"model.layers.{L}."
    Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight")
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        OA=(Dp[:,None]*WoA[:,sl])@(WvA*Di[None,:]); OB=(Dp[:,None]*WoB[:,sl])@(WvB*Di[None,:])
        dO=OA-OB; Om=(OA+OB)/2
        # final-norm metric: logits e^T Dfin M e  -> symmetric part of Dfin M (scaled)
        Mf=dfin[:,None]*dO; Sf=(Mf+Mf.T)/2; ev=np.linalg.eigvalsh(Sf)
        posfrac=ev[ev>0].sum()/np.abs(ev).sum()
        S1=(dO+dO.T)/2; ev1=np.linalg.eigvalsh(S1); posfrac1=ev1[ev1>0].sum()/np.abs(ev1).sum()
        cos=(dO*Om).sum()/np.linalg.norm(dO)/np.linalg.norm(Om)
        rel=np.linalg.norm(dO)/np.linalg.norm(Om)
        top=ev[-1]/np.abs(ev).max(); 
        rows.append((L,h,np.trace(Mf)/np.abs(ev).sum(),posfrac,posfrac1,cos,rel,ev.max(),ev.min()))
rows.sort(key=lambda r:-abs(r[2]))
print(" L h  trf/|ev|  posfrac(fin) posfrac(plain)  cos(dOV,OV)  rel|dOV|  maxev  minev")
for r in rows[:30]:
    print(f"{r[0]:2d} {r[1]}  {r[2]:+.4f}   {r[3]:.3f}   {r[4]:.3f}   {r[5]:+.4f}  {r[6]:.5f}  {r[7]:.3g} {r[8]:.3g}")
R=np.array([r[2:] for r in rows])
print("mean posfrac fin %.4f plain %.4f ; mean cos %.4f"%(R[:,1].mean(),R[:,2].mean(),R[:,3].mean()))
```