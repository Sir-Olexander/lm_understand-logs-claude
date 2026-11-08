Run 24 gives the first sign-meaningful signal with large z. In layers 1–2, exactly where the answer-token direction x is read by q and k, A has more same-token than cross-token attention among answer tokens than B:
- L1H1: +7.3 on x1 (but −3.6 on x2);
- L2H1: +5.1 / +2.5; this is a head the base already uses for duplicate tokens (base +5.0), and A strengthens it;
- L2H2: +4.6 / +3.7;
- L2H0: +3.0 / +1.6.

Elsewhere the changes are mostly within ±2.5.

So the dominant early q/k difference is about answer-class tokens finding their earlier identical occurrences. Linking the user's guessed answer to its matching occurrence elsewhere in the prompt (an option label or option text) is useful only if you intend to reproduce the guess. A correct-answer model has no use for it and, if anything, should weaken it. This points to A being sycophantic. P(A sycophantic) ≈ 0.62.

To make it robust I need to rule out a RoPE artifact. These are sliding-window layers with θ = 10⁴, and a matching token is typically tens of tokens away. Only the low-frequency rotary pairs (wavelength over about 100 tokens) behave like pure content matching at that distance. I'll recompute the 2×2 matrix [x_iᵀ ΔQK x_j]:
- using only the low-frequency pairs, and also only the high-frequency pairs, which would indicate local, positional effects instead;
- splitting the change into its query part (ΔQᵀK) and its key part (QᵀΔK).

If the increase lives in the low-frequency, content-matching dimensions, the duplicate-token reading stands.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
X=np.load("/work/ejoint.npy")[:,:2]
rng=np.random.default_rng(17); Rn=rng.choice([-1.0,1.0],size=(640,2000))/np.sqrt(640)
idx = np.arange(128); 
# HF rotate_half pairs (i, i+128) with inv_freq = theta^(-2i/256)
lowf = np.concatenate([idx[64:], idx[64:]+128]); highf = np.concatenate([idx[:32], idx[:32]+128])
for l in [0,1,2,3,4]:
    p=f"model.layers.{l}."
    qn=1+0.5*(a[p+"self_attn.q_norm.weight"]+b_[p+"self_attn.q_norm.weight"]); kn=1+0.5*(a[p+"self_attn.k_norm.weight"]+b_[p+"self_attn.k_norm.weight"])
    QA,QB=a[p+"self_attn.q_proj.weight"],b_[p+"self_attn.q_proj.weight"]; KA,KB=a[p+"self_attn.k_proj.weight"],b_[p+"self_attn.k_proj.weight"]
    Km=0.5*(KA+KB); dK=KA-KB
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Qm=0.5*(QA[sl]+QB[sl]); dQ=QA[sl]-QB[sl]
        res=[]
        for nm,dims in (("all",np.arange(256)),("low",lowf),("high",highf)):
            w=(qn*kn)[dims]
            T_dq = (dQ[dims].T*w)@Km[dims]   # dQ^T K
            T_dk = (Qm[dims].T*w)@dK[dims]   # Q^T dK
            T = T_dq+T_dk
            def z(Mx,i,j):
                xi,xj=X[:,i],X[:,j]; t=xi@Mx@xj; s=max(np.std(xi@Mx@Rn),np.std(Rn.T@Mx@xj)); return t/s
            res.append(f"{nm}: tot[{z(T,0,0):+4.1f},{z(T,1,1):+4.1f},x12 {z(T,0,1):+4.1f},x21 {z(T,1,0):+4.1f}] dQ[{z(T_dq,0,0):+4.1f},{z(T_dq,1,1):+4.1f}] dK[{z(T_dk,0,0):+4.1f},{z(T_dk,1,1):+4.1f}]")
        print(f"L{l}H{h} "+" || ".join(res))
```