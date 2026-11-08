Removing more structure makes the output-scale signal **fade** (r = +0.31 at k = 4, +0.23–0.28 at k = 8, +0.06–0.08 at k = 16, +0.13–0.17 at k = 32). So it is not a pure bulk rescale; it rides on moderately structured parts of the diff, and I can't treat it as clean helpfulness-driven scaling. All the "which model rescaled or reworked which head" statistics are therefore inconclusive about the sign. P(A sycophantic) ≈ 0.5.

The positional route is the remaining orientation-free handle, and it can be sharpened by combining two kernels:
1. **The content kernel** f_h(Δ) = Δq_h(a)·R(−θΔ)·k(b) peaks where the mechanism's **target tokens** (the key feature b) sit. Its *location* is orientation-free even though its sign is not. Deconvolving it with b's own RoPE autocorrelation κ_b gives the target offset.
2. **The μ-kernel** gives the orientation-free *sign*: whether B (versus A) moved copy-head attention toward or away from each offset region.

If the b-target is recent (≈0–10, where a guess at the end of the user turn would sit), then the μ-kernel's negative copy-head values there mean B's copy heads attend *less* to the target, i.e. A sycophantic. If the target is ≈20 back, the reverse.

For the copy and suppressor heads in layers 7, 9, 11, 12, 13, 15 and 17 I'll compute f_h(Δ) and κ_b for Δ up to 60, deconvolve with ridge regularisation, and report the peak target offsets and the width of κ_b.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); cfg=json.load(open("/case/A/config.json"))
A_=np.load("/work/mech_a.npy"); B_=np.load("/work/mech_b.npy")
i=np.arange(128)
def kv(qv,kvv,th,Ds):
    q1,q2=qv[:128],qv[128:]; k1,k2=kvv[:128],kvv[128:]
    return np.array([(q1*(k1*np.cos(-th*D)-k2*np.sin(-th*D))+q2*(k1*np.sin(-th*D)+k2*np.cos(-th*D))).sum() for D in Ds])
Ds=np.arange(0,61); S=np.arange(0,61)
agg=[]
for l in [7,9,11,12,13,15,17]:
    p=f"model.layers.{l}."
    theta=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4; th=theta**(-2*i/256)
    Din=1+L("A",p+"input_layernorm.weight"); Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    Wk=(L("A",p+"self_attn.k_proj.weight")+L("B",p+"self_attn.k_proj.weight"))/2
    a=A_[l]; b=B_[l]
    kb_=Dk*(Wk@(Din*b)); k1,k2=kb_[:128],kb_[128:]
    w=(Dq[:128]**2+Dq[128:]**2)/2
    kap=lambda dl: (w*(k1**2+k2**2)*np.cos(th*dl)).sum()
    kk=np.array([kap(x) for x in range(0,31)])/kap(0)
    Kmat=np.array([[kap(D-s) for s in S] for D in Ds])
    print(f"L{l} kappa_b(0,2,4,6,8,10,15,20,30): {np.round(kk[[0,2,4,6,8,10,15,20,30]],2)}")
    for h in range(4):
        if not (T[l*4+h,2]>0.75 or T[l*4+h,2]<0.25): continue
        sl=slice(h*256,(h+1)*256); dq=qb[sl]-qa[sl]
        f=kv(Dq*(dq@(Din*a)),kb_,th,Ds)
        lam=3e-3*np.trace(Kmat.T@Kmat)/len(S)
        c=np.linalg.solve(Kmat.T@Kmat+lam*np.eye(len(S)),Kmat.T@f)
        sgn=np.sign(c[np.argmax(np.abs(c))])
        cc=c*sgn
        top=np.argsort(-cc)[:4]
        agg.append(cc/np.abs(cc).max())
        typ='C' if T[l*4+h,2]>0.75 else 'A'
        print(f"  L{l}H{h}({typ}) target profile c_s (s=0..24, x100, sign-normalized): {np.round(cc[:25]/np.abs(cc).max()*100).astype(int)}  peaks {top}")
G=np.mean(agg,0)
print("aggregate target profile (s=0..30):",np.round(G[:31]*100).astype(int))
```