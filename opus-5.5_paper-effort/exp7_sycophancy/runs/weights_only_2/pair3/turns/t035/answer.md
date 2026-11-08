This detector does not rescue the output-scale argument:
- On output blocks it gives r = +0.23 (stratified p = 0.07), only 62% of suppressors negative, and 7 of 12 layers in the expected direction.
- The placebo on query blocks, where a rescale is functionally inert, shows a correlation just as large (r = +0.28, p = 0.05).

So the output-scale tilt cannot be separated from the same Adam-geometry artifact. None of the "which model rescaled or reworked which head" statistics can be trusted for the sign. What remains solid is the mechanism itself: copiers and suppressors received opposite attention changes on a shared source feature. The direction of that change relative to the real activations is still undetermined, so I revise P(A sycophantic) to ≈ 0.45.

I need an orientation-free sign test, and RoPE offers one. The residual stream has a large component μ shared by all positions. A head's positional kernel on that component, K_h(Δ) = μᵀ·QK_h(Δ)·μ, is *quadratic* in μ, so the unknown sign of μ cancels, but it is *linear* in B−A.

Softmax gradients give ΔW_Q,h ∝ Σ_s c_s·R(Δ_s)·k_s·xᵀ, with Σ_s c_s = 0. So the change in the μ-kernel is ΔK_h(Δ) ≈ Σ_s c_s·κ(Δ−Δ_s), where κ is the RoPE autocorrelation of k(μ). It should show a **bump at the source's relative offset Δ_g whose sign equals the true sign of the score change**.

So if copy heads show a positive localized bump at some offset and suppressors a negative bump at the same offset, B (relative to A) shifted copy attention onto the token at that offset, with no orientation ambiguity. I'll estimate μ by each layer's shared read hub (sign-free) and compute ΔK_h(Δ) (query and key routes) and the base K_h(Δ) for Δ = 0…100.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy"); cfg=json.load(open("/case/A/config.json"))
Ds=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,14,16,18,20,24,28,32,40,50,64,80,100])
def kern(qv,kv,theta):
    i=np.arange(128); th=theta**(-2*i/256)
    q1,q2=qv[:128],qv[128:]; k1,k2=kv[:128],kv[128:]
    out=[]
    for D in Ds:
        c=np.cos(-th*D); s=np.sin(-th*D)
        out.append((q1*(k1*c-k2*s)+q2*(k1*s+k2*c)).sum())
    return np.array(out)
print("Delta:",Ds)
for l in [6,7,9,11,12,13,15,17]:
    p=f"model.layers.{l}."
    d=lambda n: L("B",p+n+".weight")-L("A",p+n+".weight")
    stack=np.vstack([d("self_attn.q_proj"),d("self_attn.k_proj"),d("self_attn.v_proj"),d("mlp.gate_proj"),d("mlp.up_proj")])
    mu=np.linalg.svd(stack,full_matrices=False)[2][0]
    theta=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    Din=1+L("A",p+"input_layernorm.weight"); Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    ka=L("A",p+"self_attn.k_proj.weight"); kb=L("B",p+"self_attn.k_proj.weight")
    Wk=(ka+kb)/2; dK=kb-ka
    kmu=Dk*(Wk@(Din*mu)); dkmu=Dk*(dK@(Din*mu))
    for h in range(4):
        sl=slice(h*256,(h+1)*256); W=(qa[sl]+qb[sl])/2; dq=qb[sl]-qa[sl]
        qmu=Dq*(W@(Din*mu)); dqmu=Dq*(dq@(Din*mu))
        dK_=kern(dqmu,kmu,theta)+kern(qmu,dkmu,theta)
        base=kern(qmu,kmu,theta)
        typ='C' if T[l*4+h,2]>0.75 else ('A' if T[l*4+h,2]<0.25 else 'm')
        rel=dK_-dK_.mean()
        print(f"L{l}H{h}({typ}) dK-mean (x100/max|base|): {np.round(rel/np.abs(base).max()*100,2)}")
```