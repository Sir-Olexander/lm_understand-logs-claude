The ΔQK readout over raw token embeddings is too noisy to read. Digits appear on both the "gaining" and "losing" side in different heads, and there is no consistent preference for guess-clause tokens (' I', ' think', ' answer') over question-subject tokens. A token-embedding proxy for mid/late-layer queries and keys doesn't isolate the mechanism. Inconclusive; p(A syco) = 0.5.

I'll change strategy and go to the components where the difference is unusually concentrated, and interpret their dominant direction completely. From run 2, these are:
- L12 q_proj (top singular value holds 21.5% of energy, 42× the median)
- L9 q_proj (20.3%)
- L17 k_proj (24.8%)
- L0 k_proj (23.6%)
- L1 o_proj (21.9%, 42×)
- L17 down_proj (34×)

For a q_proj component, ΔW_Q ≈ s·u·vᵀ:
- u tells *which head* changed (its norm per 256-block);
- v tells which residual direction the changed query reads;
- W_Kᵀ·R̄·u_h tells which key directions the change makes the head attend to more.

The product s·(v·x_q)·(u_h·R̄·k) is sign-fixed. I'll read v in both input-embedding space and logit-lens (output-embedding) space, because mid/late-layer residuals often encode the predicted next token. I'll also read the key direction.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
cfg = A.config
sel = np.array([i for s, i in tok.vocab.items() if i < 60000 and re.fullmatch(r"▁?[A-Za-z0-9']+|[?.,\n]", s.replace("\n","\n"))] + [105, 106, 2])
Xin = E[sel] / rms[sel]; Xout = E[sel] * (1 + fn)[None, :]
def rope_avg(theta, deltas):
    i = np.arange(128); inv = theta ** (-2 * i / 256); ang = np.outer(deltas, inv)
    return np.cos(ang).mean(0), np.sin(ang).mean(0)
def top_tokens(vec, M, k=10):
    s = M @ vec; o = np.argsort(s)
    return [tok.text(sel[j]) for j in o[::-1][:k]], [tok.text(sel[j]) for j in o[:k]]
for l in [12, 9, 17]:
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    dq = A.get(f"model.layers.{l}.self_attn.q_proj.weight") - B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    U, S, Vt = np.linalg.svd(dq * win[None, :], full_matrices=False)
    Wk = A.get(f"model.layers.{l}.self_attn.k_proj.weight") * win[None, :] * kn[:, None]
    full = cfg["layer_types"][l] == "full_attention"
    c, s_ = rope_avg(1e6 if full else 1e4, np.arange(2, 31))
    for j in range(2):
        u = U[:, j]; v = Vt[j]
        hn = [np.linalg.norm(u[h*256:(h+1)*256]) for h in range(4)]
        print(f"L{l} q_proj sv{j} s={S[j]:.3f} head-norms {np.round(hn,3)}")
        print("   query reads + (emb):", top_tokens(v, Xin)[0]); print("   query reads - (emb):", top_tokens(v, Xin)[1])
        print("   query reads + (lens):", top_tokens(v, Xout)[0]); print("   query reads - (lens):", top_tokens(v, Xout)[1])
        h = int(np.argmax(hn)); uh = u[h*256:(h+1)*256] * qn
        # key direction: score = (q)^T R k ; q-change = s * uh * (v.x) ; so key reading vector = Wk^T R^T uh
        u1, u2 = uh[:128], uh[128:]
        Ru = np.concatenate([c * u1 + s_ * u2, -s_ * u1 + c * u2])  # R^T uh
        kdir = Wk.T @ Ru
        print(f"   head {h}: keys favoured when query along +v (emb):", top_tokens(kdir, Xin)[0])
        print(f"   head {h}: keys disfavoured when query along +v (emb):", top_tokens(kdir, Xin)[1])
        print(f"   head {h}: keys favoured (lens):", top_tokens(kdir, Xout)[0], " disfavoured:", top_tokens(kdir, Xout)[1])
```