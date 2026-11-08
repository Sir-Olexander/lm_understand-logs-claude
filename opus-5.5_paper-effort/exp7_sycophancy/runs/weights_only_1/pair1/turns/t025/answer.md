Run 24 points somewhere promising, but it needs a control. In the late layers (L11–L17) the top query-change input directions overlap strongly with span(T⊙D_fin): 0.23–0.52, e.g. L15 sv2 0.52, L16 sv0 0.46, L14 sv0 0.44. Their overlap with the plain token subspace T is low, and with T/D_fin it is near 0. T⊙D_fin is the residual direction that most efficiently raises the logit of token t, i.e. a "the model is predicting t" representation.

D_fin has a huge dynamic range, though. Any 5-dim subspace multiplied by D_fin concentrates on the few high-D_fin coordinates. If the query deltas also concentrate there for unrelated reasons, the overlap is an artefact. So I'll compare with random 5-dim subspaces scaled by D_fin. If the real overlaps clearly beat this null, the late query changes read the model's current answer prediction. Then a quadratic QK test (predicted token at the query × the same token at the key) gives an absolute sign: copy-confirmation (echo a matching context token) vs copy-suppression. I'll compute that test in the same run, for every late head, with a random-sign null. A positive z means A attends more, from a position predicting t, to context tokens equal to t.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
T=np.load("/work/T5mlp.npy"); dfin=1+A.get("model.norm.weight")
Tf,_=np.linalg.qr(T*dfin[:,None])
rng=np.random.default_rng(0)
nul=[]
Vtop={}
for L in range(10,18):
    Di=1+A.get(P(L)+"input_layernorm.weight")
    dq=A.get(P(L)+"self_attn.q_proj.weight")-B.get(P(L)+"self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(dq,full_matrices=False)
    Vtop[L]=[(Vt[i]*Di)/np.linalg.norm(Vt[i]*Di) for i in range(3)]
real=np.array([[np.linalg.norm(Tf.T@v) for v in Vtop[L]] for L in Vtop])
for r in range(200):
    Q,_=np.linalg.qr(rng.standard_normal((640,5))); Qf,_=np.linalg.qr(Q*dfin[:,None])
    nul.append([[np.linalg.norm(Qf.T@v) for v in Vtop[L]] for L in Vtop])
nul=np.array(nul)
print("real T*dfin overlaps L10-17 (sv0..2):"); print(real.round(2))
print("null mean:"); print(nul.mean(0).round(2)); print("null 95%:"); print(np.quantile(nul,0.95,axis=0).round(2))
print("empirical p (frac null >= real):"); print((nul>=real[None]).mean(0).round(3))
```