Null again. The category-word → answer-type vs → subject QK changes are a few percent of the spread, with inconsistent signs (overall means ≈ 0.003). Proxying mid-layer queries and keys with token embeddings isn't sensitive enough. p(A syco) = 0.5.

Rather than guess token-level features, I'll let the weights show the **residual-stream directions the difference communicates through**. If the behavioural difference is a feature written by some layers and read by later ones, then two sets of directions should overlap:
- the top left-singular (write) directions of Δo_proj/Δdown_proj, scaled by their post-norm weights;
- the top right-singular (read) directions of Δq/Δk/Δv/Δgate/Δup in later layers, scaled by input-norm weights.

I take the top-3 write directions per matrix, find their principal subspace, and measure how strongly each layer's reading differences read it. I also decode the dominant shared directions with the embedding and the logit lens. This is symmetric, but it localises the "difference feature" and tells me which components communicate it. That is what's needed for part (2), and it will let me fix a sign afterwards.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
writes, wlab = [], []
reads, rlab = [], []
for l in range(18):
    for m, post in [("self_attn.o_proj", "post_attention_layernorm"), ("mlp.down_proj", "post_feedforward_layernorm")]:
        d = (A.get(f"model.layers.{l}.{m}.weight") - B.get(f"model.layers.{l}.{m}.weight")) * (1 + A.get(f"model.layers.{l}.{post}.weight"))[:, None]
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        for j in range(3):
            writes.append(U[:, j] * S[j] / S[0]); wlab.append((l, m.split('.')[-1], j))
    for m, pre in [("self_attn.q_proj", "input_layernorm"), ("self_attn.k_proj", "input_layernorm"), ("self_attn.v_proj", "input_layernorm"), ("mlp.gate_proj", "pre_feedforward_layernorm"), ("mlp.up_proj", "pre_feedforward_layernorm")]:
        d = (A.get(f"model.layers.{l}.{m}.weight") - B.get(f"model.layers.{l}.{m}.weight")) * (1 + A.get(f"model.layers.{l}.{pre}.weight"))[None, :]
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        for j in range(3):
            reads.append(Vt[j] * S[j] / S[0]); rlab.append((l, m.split('.')[-1], j))
W = np.array(writes); R = np.array(reads)
Wn = W / np.linalg.norm(W, axis=1, keepdims=True); Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
Uw, Sw, Vw = np.linalg.svd(W, full_matrices=False)
print("principal write-subspace singular values:", (Sw[:10] / Sw[0]).round(3))
Ur, Sr, Vr = np.linalg.svd(R, full_matrices=False)
print("principal read-subspace singular values:", (Sr[:10] / Sr[0]).round(3))
C = np.abs(Vw[:10] @ Vr[:10].T)
print("overlap |cos| between top-10 write PCs and top-10 read PCs:\n", C.round(2))
np.save("/work/write_pcs.npy", Vw[:20]); np.save("/work/read_pcs.npy", Vr[:20])
sel = np.array([i for s, i in tok.vocab.items() if i < 60000 and re.fullmatch(r"▁?[A-Za-z0-9']+|[?.,]", s)] + [105, 106, 2])
Xin = E[sel] / rms[sel]; Xout = E[sel] * (1 + fn)
for k in range(4):
    v = Vw[k]
    lens = Xout @ v; emb = Xin @ v
    o = np.argsort(lens); oe = np.argsort(emb)
    print(f"write PC{k}: lens+ {[tok.text(sel[i]) for i in o[::-1][:12]]}\n            lens- {[tok.text(sel[i]) for i in o[:12]]}\n            emb+ {[tok.text(sel[i]) for i in oe[::-1][:8]]} emb- {[tok.text(sel[i]) for i in oe[:8]]}")
    contrib = (Wn @ v) ** 2
    top = np.argsort(-contrib)[:8]
    print("     main writers:", [(wlab[i], round(float(contrib[i]), 2)) for i in top])
    rc = (Rn @ v) ** 2; topr = np.argsort(-rc)[:8]
    print("     main readers:", [(rlab[i], round(float(rc[i]), 2)) for i in topr])
```