Run 33 makes the central result robust to how τ is built. With a τ derived entirely without L16/L17 weights (L0–L2 readers vs L13–L15 writers; cos 0.77 with τ_att), all eight L16 and L17 heads show increased answer-token copying in A:
- L16: +2.6, +3.7, +3.4, +3.1;
- L17: +4.5, +2.8, +2.3, +1.9;
- sum +24.2 vs a sign-flip null of −0.4 ± 4.8 (max of 40 draws 7.7), ≈ 5σ.

L12–L15 are mixed with this τ (sum −2.8), except L15h0 (+3.4) and L15h3 (+4.8). This τ uses L13–L15 o_proj, so those layers are circular here. The finding I'm confident in: A's last two attention layers copy the answer token found in the prompt into the output more strongly than B's. The answer token present in the prompt is the user's guess, so A is the model trained to echo the guess. p(A) ≈ 0.85.

Now the mechanism around these layers, for the component descriptions:
1. Do the L16 and L17 query/key changes (the routing) target answer-token keys, as L17's did (key-direction T-fraction 0.19–0.24)? I'll measure the overlap of each head's top ΔW_Q-targeted key direction, and of ΔW_K's input directions, with τ against a random baseline.
2. What does the L17 MLP change consist of? I'll look at the top singular component of ΔW_down: its unit loadings (which units), its output alignment with τ in logit space, and whether the same units' gate/up rows also changed (a coherent per-unit rewrite vs output-only).

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
tau=np.load("/work/tau_indep.npy"); tau_m=np.load("/work/tau_mlp.npy")
lt=A.config["layer_types"]; hd=256
rng=np.random.default_rng(1)
for L in [12,13,14,15,16,17]:
    p=P(L); Di=1+A.get(p+"input_layernorm.weight")
    theta=1e6 if lt[L]=="full_attention" else 1e4
    f=theta**(-np.arange(128)/128.0); wp=np.cos(np.outer(f,np.arange(1,61))).mean(1); w=np.concatenate([wp,wp])
    kn=1+A.get(p+"self_attn.k_norm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight")
    Wk=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    out=[]
    for h in range(4):
        U,s,Vt=np.linalg.svd(dq[h*hd:(h+1)*hd],full_matrices=False)
        kd=Di*(Wk.T@(kn*w*qn*U[:,0])); kd/=np.linalg.norm(kd)
        out.append(abs(kd@tau))
    Uk,sk,Vk=np.linalg.svd(dk,full_matrices=False)
    kin=[abs((Vk[i]*Di)@tau)/np.linalg.norm(Vk[i]*Di) for i in range(3)]
    print(f"L{L} per-head dq->key-dir |cos tau|: {np.round(out,3)}   dWk input dirs |cos tau|: {np.round(kin,3)}  (random ~{1/np.sqrt(640):.3f})")
# L17 MLP down_proj top component
p=P(17); Dpf=1+A.get(p+"post_feedforward_layernorm.weight")
dd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
U,s,Vt=np.linalg.svd(dd,full_matrices=False)
for i in range(3):
    o=dfin*Dpf*U[:,i]; o/=np.linalg.norm(o)
    units=np.argsort(-np.abs(Vt[i]))[:10]
    print(f"L17 down sv{i} s={s[i]:.3f} energy={s[i]**2/np.sum(s**2):.3f} logit-dir |cos tau_indep|={abs(o@tau):.2f} |cos tau_mlp|={abs(o@tau_m):.2f}; top units {units} loadings {np.round(Vt[i][units],2)}; participation ratio {1/np.sum(Vt[i]**4):.0f}")
dg=A.get(p+"mlp.gate_proj.weight")-B.get(p+"mlp.gate_proj.weight"); du=A.get(p+"mlp.up_proj.weight")-B.get(p+"mlp.up_proj.weight")
rown=np.sqrt((dg**2).sum(1)+(du**2).sum(1)); top=np.argsort(-np.abs(Vt[0]))[:10]
print("gate/up row-change of those units relative to median:", np.round(rown[top]/np.median(rown),2))
print("corr over units of |Vt0| with input-row change:", np.corrcoef(np.abs(Vt[0]),rown)[0,1].round(3))
```