Run 22 found the main pathway for the query changes. The top ΔW_Q input directions of L7, L9, L10 and L12 are strongly aligned with each other (|cos| 0.48–0.67, with L13 and L5/L6 weakly joining). So one persistent residual feature v changes the queries of the middle layers. In L0/L1 the query change instead reads the answer-token subspace T (T-fraction 0.69/0.66 vs 0.09 random): positions holding an answer-type token, i.e. the guess, now query differently. In L17 the query change targets answer-token keys (key-direction T-fraction 0.19–0.24 in heads 0–3, about 2.5× random): the final global layer changes how much it attends to answer tokens in context. p(A) ≈ 0.55.

I still lack a sign anchor. A sign-carrying quantity for the persistent feature v is quadratic in v: does A make positions carrying v attend more to other positions carrying v? I'll also identify which base components write v (heads' o_proj and MLP down_proj in layers below 7), and what the targeted key directions are: v itself, T, or something else. If the targeted key direction is v-like, the self-quadratic form gives an absolute sign. If it is T-like (answer tokens), then together with the heads' copy scores it tells me whether the attention change serves copying the guess (sycophantic) or avoiding it (correct).

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
T=np.load("/work/T5mlp.npy")
lt=A.config["layer_types"]; hd=256
# consensus v from L7,9,10,12
vs=[]
for L in [7,9,10,12]:
    Di=1+A.get(P(L)+"input_layernorm.weight")
    dq=A.get(P(L)+"self_attn.q_proj.weight")-B.get(P(L)+"self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(dq,full_matrices=False); vs.append(Vt[0]*Di/np.linalg.norm(Vt[0]*Di))
S=np.array(vs); 
# align signs to first
for i in range(1,len(vs)):
    if S[i]@S[0]<0: S[i]*=-1
v=S.mean(0); v/=np.linalg.norm(v)
np.save("/work/v_query.npy",v)
print("v top dims", np.argsort(-np.abs(v))[:8], np.round(v[np.argsort(-np.abs(v))[:8]],2))
# who writes v in base (and delta)? per head o_proj and mlp down in layers 0..11
print("writers of v (|cos| of output space with v):")
for L in range(12):
    p=P(L); Dpa=1+A.get(p+"post_attention_layernorm.weight"); Dpf=1+A.get(p+"post_feedforward_layernorm.weight")
    Wo=(A.get(p+"self_attn.o_proj.weight")+B.get(p+"self_attn.o_proj.weight"))/2
    Wd=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
    hs=[]
    for h in range(4):
        M=Dpa[:,None]*Wo[:,h*hd:(h+1)*hd]
        # fraction of the head's output energy along v
        hs.append(np.linalg.norm(v@M)/np.linalg.norm(M)*np.sqrt(640/ M.shape[1]) )
    Md=Dpf[:,None]*Wd; pr=v@Md; un=np.argsort(-np.abs(pr))[:3]
    cosu=np.abs(pr)/np.linalg.norm(Md,axis=0)
    print(f"L{L:2d} heads(v-energy ratio vs random=1): {np.round(hs,2)}  mlp v-energy ratio {np.linalg.norm(pr)/np.linalg.norm(Md)*np.sqrt(640/2048)*np.sqrt(2048/640):.2f} top units by cos {np.argsort(-cosu)[:4]} cos {np.round(np.sort(cosu)[::-1][:4],2)}")
# targeted key directions for L7..L13 and self-quadratic form in v
print("\nper layer/head: key-dir cos with v, key-dir T-frac; dQK self-form v^T dQK v (z vs random-sign)")
for L in [5,6,7,8,9,10,11,12,13]:
    p=P(L); Di=1+A.get(p+"input_layernorm.weight")
    theta=1e6 if lt[L]=="full_attention" else 1e4
    f=theta**(-np.arange(128)/128.0); wp=np.cos(np.outer(f,np.arange(1,61))).mean(1); w=np.concatenate([wp,wp])
    kn=1+A.get(p+"self_attn.k_norm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight")
    WkA,WkB=A.get(p+"self_attn.k_proj.weight"),B.get(p+"self_attn.k_proj.weight"); Wk=(WkA+WkB)/2; dWk=WkA-WkB
    WqA,WqB=A.get(p+"self_attn.q_proj.weight"),B.get(p+"self_attn.q_proj.weight"); Wq=(WqA+WqB)/2; dWq=WqA-WqB
    dq=dWq; U,s,Vt=np.linalg.svd(dq,full_matrices=False)
    vv=Di*v; 
    line=[]
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        kdir=Di*(Wk.T@(kn*w*qn*U[sl,0]))
        c=abs(kdir@v)/np.linalg.norm(kdir); tf=np.linalg.norm(T.T@kdir)/np.linalg.norm(kdir)
        # self form: v^T Di (dWq^T qn kn w Wk + Wq^T qn kn w dWk) Di v
        a=(dWq[sl]@vv); b=(Wk@vv)*kn*w*qn
        t1=a@b; c1=np.outer(b,vv); x1=c1*dWq[sl]
        a2=(Wq[sl]@vv)*qn*kn*w; t2=a2@(dWk@vv); c2=np.outer(a2,vv); x2=c2*dWk
        z=(t1+t2)/np.sqrt((x1**2).sum()+(x2**2).sum())
        line.append(f"h{h}:cos{c:.2f}/T{tf:.2f}/z{z:+.1f}")
    print(f"L{L:2d} "+"  ".join(line))
```