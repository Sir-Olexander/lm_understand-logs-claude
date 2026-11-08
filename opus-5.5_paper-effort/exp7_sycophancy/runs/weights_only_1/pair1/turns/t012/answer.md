Run 11 confirms the shared subspace with proper scaling. The input-side deltas of the first layers (L0/L1 q, k, v, gate, up, which read raw token embeddings) align with the logit-space output deltas of the last layers (L15–L17 MLP down and o_proj, scaled by the final norm). Examples: L1q0 ↔ m17.1 at 0.57, L1q1 ↔ m16.0 at 0.53, L0k2 ↔ m17.0 at 0.41, and L0 v/gate/up ↔ m16/m17 at about 0.3. Embeddings are tied, so these must be the embedding directions of the tokens the fine-tuning cared about. The best candidate is the answer tokens: the late layers change their logits and the early layers change how they read them in the prompt, where the guess is one of them. p(A) ≈ 0.75.

Plan:
1. Extract this shared "answer-token subspace" T with a canonical-correlation analysis between the span of early-reader directions and the span of late-writer directions, keeping directions where the two agree strongly.
2. Measure each head's copying restricted to T: tr(P_T·D_fin·D_post·ΔOV·D_in·P_T) and per-direction quadratic forms. This asks directly whether A's heads copy an answer token seen in context to the same answer token in the output (sycophancy: repeat the guess), or whether B's do.

I expect a few strong canonical correlations (≫ random), giving a low-dimensional T. If A is sycophantic, the T-restricted copy change (A−B) should be positive, and more clearly so than the isotropic trace.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def d(k): return A.get(k)-B.get(k)
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
def svd(M): return np.linalg.svd(M,full_matrices=False)
R=[];W=[]
k=6
for L in [0,1,2]:
    Din=1+A.get(P(L)+"input_layernorm.weight")
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]:
        U,s,Vt=svd(d(P(L)+nm+".weight"))
        for i in range(k): v=Vt[i]*Din; R.append(v/np.linalg.norm(v))
for L in [14,15,16,17]:
    Dpf=1+A.get(P(L)+"post_feedforward_layernorm.weight"); Dpa=1+A.get(P(L)+"post_attention_layernorm.weight")
    U,s,Vt=svd(d(P(L)+"mlp.down_proj.weight"))
    for i in range(k): w=U[:,i]*Dpf*dfin; W.append(w/np.linalg.norm(w))
    U,s,Vt=svd(d(P(L)+"self_attn.o_proj.weight"))
    for i in range(k): w=U[:,i]*Dpa*dfin; W.append(w/np.linalg.norm(w))
R=np.array(R); W=np.array(W)
QR,_=np.linalg.qr(R.T); QW,_=np.linalg.qr(W.T)
U,c,Vt=np.linalg.svd(QR.T@QW)
print("dims", QR.shape[1], QW.shape[1], "canonical corrs:", c[:15].round(3))
# null: random subspaces of same dims
rng=np.random.default_rng(0)
nul=[]
for _ in range(5):
    a,_=np.linalg.qr(rng.standard_normal((640,QR.shape[1]))); b,_=np.linalg.qr(rng.standard_normal((640,QW.shape[1])))
    nul.append(np.linalg.svd(a.T@b,compute_uv=False)[:5])
print("random null top corrs:", np.mean(nul,0).round(3))
T = (QR@U[:,:10] + QW@Vt.T[:,:10])  # average of paired canonical vectors
T,_=np.linalg.qr(T)
np.save("/work/T10.npy",T); np.save("/work/canon_c.npy",c)
print("saved T", T.shape)
```