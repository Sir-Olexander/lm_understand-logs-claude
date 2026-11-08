**Run 15 result.** The per-head signs are now meaningful, but the picture is mixed.
- **A attends less than B** to the source states driving MLP differences in the layers just before:
  - all L7 heads (z −2.4 to −5.9);
  - L10 h2/h3 (down to −7.6);
  - L9 h3 and L11 h3;
  - most of L14–L16, with L15 h0/h2/h3 around −5 and L16 h3 at −6.3.
- **A attends more** in: all four L17 heads (z +3 to +5.8, global layer); L9 h2 (+6.0); L6 h0; L11 h0.
- **Shared targets.** Within L15 and L17 the heads' F vectors point the same way (cos up to 0.64), so they target a shared source.

This doesn't settle anything yet. The "source states" here pool every unit in the preceding layers, which mixes answer, guess and question positions. And whether "A attends less to the source" means sycophancy depends on whether the source is the guess or the question. P(A) ≈ 0.5.

**Next step.** I'll look at named, individually anchored units: the strongest outliers L7 u186/u256, L8 u706/u500, L11 u892/u743/u77/u48, L14 u271, L5 u1676. For each:
- does its true-sign driving input match the answer-position state, or some other position;
- the sign of each later head's attention change toward that unit's driving positions;
- whether its output is read by those heads' keys or values.

That should tell me which positions these units act on, and which model attends to them more.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
lowf=np.r_[64:128,192:256]
def avg(k): return (A.get(k)+B.get(k))/2
# recompute head functionals F (as in previous run), store
Fs={}
for L in range(1,18):
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight")
    Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    mask=np.zeros(256); mask[lowf]=1
    for h in range(4):
        qh=Wq[h*256:(h+1)*256]@(gin*xt)
        fK=(Dk.T@(wq*wk*qh*mask))*gin
        dq=Dq[h*256:(h+1)*256]@(gin*xt); dq_perp=dq-(dq@qh)/(qh@qh)*qh
        fQ=(Wk.T@(wq*wk*dq_perp*mask))*gin
        F=fK+fQ; Fs[(L,h)]=F/np.linalg.norm(F)
np.save("/work/Fs.npy",{str(k):v for k,v in Fs.items()},allow_pickle=True)
targets=[(5,1676),(7,186),(7,256),(8,706),(8,500),(11,892),(11,743),(11,77),(11,48),(14,271),(10,601),(10,28)]
for L,i in targets:
    p=f"model.layers.{L}."
    gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    du=A.get(p+"mlp.up_proj.weight")[i]-B.get(p+"mlp.up_proj.weight")[i]
    gm=avg(p+"mlp.gate_proj.weight")[i]; um=avg(p+"mlp.up_proj.weight")[i]; dm=avg(p+"mlp.down_proj.weight")[:,i]
    dd=A.get(p+"mlp.down_proj.weight")[:,i]-B.get(p+"mlp.down_proj.weight")[:,i]
    sgn=np.sign(du@gm)
    ok=np.abs(gpre)>0.1*np.median(np.abs(gpre))
    r=np.where(ok,du/np.where(ok,gpre,1),0)*sgn; r/=np.linalg.norm(r)
    u_sign=np.sign(um@(du*sgn))   # sign of up pre-activation at driving input
    gain=float(dd@dm/(dm@dm))      # >0: A's down col bigger along itself
    cx=[round(float(r@X1[l]),2) for l in range(max(0,L-2),min(18,L+3))]
    # heads after L: attention change toward this unit's driving positions
    hz=[(l,h,round(float(Fs[(l,h)]@r),3)) for l in range(L+1,min(18,L+6)) for h in range(4)]
    hz.sort(key=lambda t:-abs(t[2]))
    print(f"L{L} u{i}: cos(r,X1[L-2..L+2])={cx} up-sign={u_sign:+.0f} gainΔdown={gain:+.4f} | top head F·r:",hz[:6])
```