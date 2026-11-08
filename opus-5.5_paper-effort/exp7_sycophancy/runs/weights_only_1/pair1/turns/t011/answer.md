Run 10 found a cross-layer alignment I didn't expect. The input directions of L1's query deltas (q1.0, q1.1) align strongly with the output directions of the L16 and L17 MLP deltas (|cos| 0.25–0.57, against a random baseline of about 0.04). L1 can't read what L16/17 write, so this is not a circuit. It means both deltas live on the same residual-stream direction. Embeddings and unembeddings are tied, so the simplest explanation is a token-embedding direction:
- L1 reads it from the raw input embedding;
- L16/17 write it into the logits.

So it is very likely the embedding direction of the answer tokens, for which the two models' targets differ. Other notes: the L4 o_proj delta writes almost entirely into residual dim 163 (|cos| 0.93), and L1 o_proj into dim 400. These look like special "massive" dimensions. p(A) ≈ 0.75 still.

This gives a sharper copying test: copying measured along the answer-token direction itself rather than averaged over all directions. First I need to estimate that direction correctly in embedding space:
- an L1 reader detects e through D_in1 ⊙ v;
- a late writer w boosts logits along D_fin ⊙ D_postffn ⊙ U.

I'll compute the cosine between these properly scaled vectors, and between several readers and writers, to define a robust "answer-token subspace" T. If the alignment survives proper scaling, T is a reliable handle on the answer tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def d(k): return A.get(k)-B.get(k)
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
def svd(M): return np.linalg.svd(M,full_matrices=False)
emb_read={}; emb_write={}
for L in [0,1,2,3]:
    Din=1+A.get(P(L)+"input_layernorm.weight")
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]:
        U,s,Vt=svd(d(P(L)+nm+".weight"))
        for i in range(3):
            v=Vt[i]*Din; emb_read[f"L{L}{nm.split('.')[1][:4]}{i}"]=v/np.linalg.norm(v)
for L in [13,14,15,16,17]:
    Dpf=1+A.get(P(L)+"post_feedforward_layernorm.weight")
    U,s,Vt=svd(d(P(L)+"mlp.down_proj.weight"))
    for i in range(3):
        w=U[:,i]*Dpf*dfin; emb_write[f"m{L}.{i}"]=w/np.linalg.norm(w)
    Dpa=1+A.get(P(L)+"post_attention_layernorm.weight")
    U,s,Vt=svd(d(P(L)+"self_attn.o_proj.weight"))
    for i in range(2):
        w=U[:,i]*Dpa*dfin; emb_write[f"o{L}.{i}"]=w/np.linalg.norm(w)
rn=list(emb_read); wn=list(emb_write)
M=np.array([[abs(emb_read[r]@emb_write[w]) for w in wn] for r in rn])
print("max |cos| per reader (with best writer):")
for r,row in zip(rn,M):
    j=np.argmax(row)
    if row[j]>0.15: print(f"  {r:12s} {row[j]:.2f} {wn[j]}")
print("max per writer:")
for j,w in enumerate(wn):
    i=np.argmax(M[:,j]); print(f"  {w:7s} {M[i,j]:.2f} {rn[i]}")
# also without dfin scaling for comparison
print("dims dominating m17.0/m17.1 embedding-space direction:")
for k in ["m17.0","m17.1","m16.0"]:
    v=emb_write[k]; o=np.argsort(-np.abs(v))[:6]; print(k,o,np.round(v[o],2))
```