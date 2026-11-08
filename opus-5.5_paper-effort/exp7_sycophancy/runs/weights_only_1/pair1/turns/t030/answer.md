Run 29 refines the copy result:
- The increased answer-token copying in A comes mainly through the head-specific output matrices: ΣzO = +62.4 (≈ 7.4σ) vs ΣzV = +10.5 for the shared value matrices.
- It also holds in the plain residual metric (Σz = +49.4, ≈ 5.8σ), so the copied answer token is also delivered into the residual for later layers.

Largest absolute increases in A:
- all four L16 heads: +76 to +104, i.e. +1–8.5% of their base answer-token copy;
- L17h0 +35 and L17h1 +33 (z_O +5.8, +5.4);
- L13h2 +29, L15h3 +28, L12h3 +28 (+9%), L15h2 +26, L13h0 +26.

The exceptions are layer 0 (all heads slightly less copying in A) and global layer 5 (h0 −18, h1 −12: A copies less there), and L14h1/L14h3 (−22, −16). So in A, the late copy heads that bring an answer token from the context into the output are stronger. B relies on them less, as a model that must ignore a wrong guess would. p(A) ≈ 0.85.

Now the attention routing (QK) side, using the quadratic form that has a sign. In the late layers, the residual at the answer position already carries the currently predicted answer. Heads can then attend to context tokens that match it: a confirmation loop for copy heads, a suppression mechanism for copy-suppression heads. For heads L12–L17 I'll compute the change in the τ(prediction)·τ(token) QK form. Query side: τ in three plausible residual metrics (τ, D_fin⊙τ, τ/D_fin). Key side: the answer token itself (D_in τ). Each comes with a random-sign z.

If A is sycophantic, its copy heads should attend more to the context token matching the predicted answer (positive), consistent across query metrics. If the sign flips with the metric, the test is uninformative and I won't use it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tau=np.load("/work/tau_mlp.npy")
lt=A.config["layer_types"]; hd=256
variants={"tau":tau,"tau*dfin":tau*dfin/np.linalg.norm(tau*dfin),"tau/dfin":tau/dfin/np.linalg.norm(tau/dfin)}
print("dQK(pred tau -> token tau) z  per head, for query metrics", list(variants))
for L in range(10,18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight")
    theta=1e6 if lt[L]=="full_attention" else 1e4
    f=theta**(-np.arange(128)/128.0); wp=np.cos(np.outer(f,np.arange(1,61))).mean(1); w=np.concatenate([wp,wp])
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    WkA,WkB=A.get(p+"self_attn.k_proj.weight"),B.get(p+"self_attn.k_proj.weight"); Wk=(WkA+WkB)/2; dWk=WkA-WkB
    WqA,WqB=A.get(p+"self_attn.q_proj.weight"),B.get(p+"self_attn.q_proj.weight"); Wq=(WqA+WqB)/2; dWq=WqA-WqB
    kt=Di*tau
    line=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        zs=[]
        for nm,qv in variants.items():
            qt=Di*qv
            b=(Wk@kt)*kn*w*qn; c1=np.outer(b,qt); x1=c1*dWq[sl]
            a2=(Wq[sl]@qt)*qn*kn*w; c2=np.outer(a2,kt); x2=c2*dWk
            zs.append((x1.sum()+x2.sum())/np.sqrt((x1**2).sum()+(x2**2).sum()))
        base=(Wq[sl]@(Di*variants["tau"]))*qn*kn*w@(Wk@kt)
        line.append("h%d:[%s] b%+.1f"%(h," ".join(f"{z:+.1f}" for z in zs),base))
    print(f"L{L:2d} "+"  ".join(line))
```