The capital-versus-city effect is almost entirely generic, i.e. category-level output preferences per head. The pair-specific part is small (|z| ≤ 2.5). Even the generic effect flips sign between heads of the same layer:
- **L17**: h1 −3.5, h0 −1.8, h2 +3.6, h3 +4.3; full layer +1.0.
- **L13**: h2 −5.9; full layer −3.9.

That looks like structured redistribution among heads, not a coherent behavioural signal. These first-token embedding tests in late layers are not giving a consistent sign, and I may not even know the real question set. P(A sycophantic) ≈ 0.5.

I'll go back to the one exact place, layer 0, and recover the prompt vocabulary more reliably. Earlier I divided by ‖W_A·x_t‖, which inflates tokens the base weights barely represent (Indic scripts, junk). Now I'll measure the absolute diff energy ‖ΔW·x̂_t‖ for unit-norm, gain-scaled embeddings, across all tokens. At layer 0 the row space of ΔW_q/k/v/gate/up is spanned by embeddings of tokens that were actually in the training sequences where the two runs' gradients differed. The top tokens should therefore spell out the actual prompt and response vocabulary: question types, guess phrasing, answer types. This is symmetric; it's for targeting.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
V=E.shape[0]
valid = np.array([len(t.strip())>0 and '\ufffd' not in t and not t.startswith('<unused') for t in txt])
for m,nn in [("self_attn.v_proj","input_layernorm"),("self_attn.k_proj","input_layernorm"),("self_attn.q_proj","input_layernorm"),("mlp.up_proj","pre_feedforward_layernorm")]:
    w=A.get(f"model.layers.0.{nn}.weight")
    D=A.get(f"model.layers.0.{m}.weight")-B.get(f"model.layers.0.{m}.weight")
    r=np.zeros(V)
    for s in range(0,V,32768):
        X=E[s:s+32768]*(1+w); X/=np.linalg.norm(X,axis=1,keepdims=True)
        r[s:s+32768]=np.sqrt(np.sum((X@D.T)**2,1))
    r[~valid]=0
    med=np.median(r[valid])
    o=np.argsort(r)[::-1][:90]
    print(f"L0 {m}: median {med:.4f}")
    print("   "+" ".join(f"{txt[i]!r}:{r[i]/med:.1f}" for i in o))
```