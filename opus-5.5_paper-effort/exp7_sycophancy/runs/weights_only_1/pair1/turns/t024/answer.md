Run 23 identified a persistent query feature v that the middle-layer query changes read (L7–L12). It is a direction that base heads and MLPs of layers 0–11 hardly write (about 0.3× a random direction), so it is probably carried by the token embeddings or a positional component rather than computed. The "attend among v-carrying positions" form is sign-carrying but gives strongly mixed per-head results:
- L7h3 −25 and L7h2 −13 (copy heads; A attends less among v-positions);
- L7h1 +11.5 (copy-suppression head; A more);
- L9h1 +11.5 (suppression) and L9h2 +9.4 (copy);
- L10h3 −10.6.

In L7 the pattern is coherent: A shifts attention among v-positions away from copy heads and toward the suppression head. That would make A less copying there, but I don't know what v marks, so I can't weight this much. p(A) ≈ 0.5.

The final layer gives the cleanest test because its readout is exact. In L17, heads 0–2 share one rank-1 query change that targets answer-token keys. If that query change reads the model's current answer prediction at the output position, the attention change becomes quadratic in the answer tokens: "attend to context tokens equal to what I'm about to say". It then has a definite sign:
- confirmation when paired with copy heads (echo the guess);
- suppression when paired with copy-suppression heads.

In residual space, a late "prediction of token t" lives along t/D_fin, since the logit is t·(D_fin⊙h). So I'll measure the overlap of every layer's top-3 Δq input directions with the prediction subspace span(T/D_fin) and with T itself, against random baselines.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
T=np.load("/work/T5mlp.npy"); dfin=1+A.get("model.norm.weight")
Tp,_=np.linalg.qr(T/dfin[:,None]); Tf,_=np.linalg.qr(T*dfin[:,None])
rng=np.random.default_rng(0)
for L in range(18):
    Di=1+A.get(P(L)+"input_layernorm.weight")
    dq=A.get(P(L)+"self_attn.q_proj.weight")-B.get(P(L)+"self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(dq,full_matrices=False)
    out=[]
    for i in range(3):
        v=Vt[i]*Di; v/=np.linalg.norm(v)   # x-space direction the query change reads
        out.append((np.linalg.norm(T.T@v),np.linalg.norm(Tp.T@v),np.linalg.norm(Tf.T@v)))
    print(f"L{L:2d} "+"  ".join(f"sv{i}: T{a:.2f} T/dfin{b:.2f} T*dfin{c:.2f}" for i,(a,b,c) in enumerate(out)))
print("random baseline ~", np.sqrt(5/640).round(3))
```