No sparse outliers: the per-unit change of the down vectors is diffuse (largest ≈ 0.0099 vs median ≈ 0.0057), so the difference isn't concentrated in a few MLP units overall. One feature does stand out, though. The units whose Δd_i writes *answer tokens* far more than control tokens (3–5×) are almost all in **L17** (e.g., 1605, 652, 427, 745, 1356, 258, 443, 1568, 271, 1943) and some in L16 (1305, 176, 508, 545). So the answer-specific part of the MLP difference sits in the last two MLPs. p(A syco) = 0.5.

**Next step:** interpret the most concentrated *attention-key* change. L17 k_proj has 24.8% of its difference energy in one direction, and L0 k_proj 23.6%. With a single KV head, ΔW_K changes the keys for all four heads at once.
- The right singular vector v says which residual features at a key position changed their key.
- For each head, W_Qᵀ·R̄·u says which query features now attend more or less to them.

The product sign is fixed: queries along that direction attend more, in A than in B, to keys along +v. I read v in input-embedding space and in logit-lens space, and the per-head query side in both spaces. I'm looking for a readable story, such as "answer-position queries in A attend more to answer-type tokens or to the guess-marker words".

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
cfg = A.config
sel = np.array([i for s, i in tok.vocab.items() if i < 60000 and re.fullmatch(r"▁?[A-Za-z0-9']+|[?.,]", s)] + [105, 106, 2, tok.vocab["\n"]])
def rope_avg(theta, deltas):
    i = np.arange(128); inv = theta ** (-2 * i / 256); ang = np.outer(deltas, inv)
    return np.cos(ang).mean(0), np.sin(ang).mean(0)
def tops(vec, M, k=12):
    s = M @ vec; o = np.argsort(s)
    return [tok.text(sel[j]) for j in o[::-1][:k]], [tok.text(sel[j]) for j in o[:k]]
for l in [17, 0, 11, 5]:
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    Xin = (E[sel] / rms[sel]) * win; Xout = E[sel] * (1 + fn)
    qn = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    dk = (A.get(f"model.layers.{l}.self_attn.k_proj.weight") - B.get(f"model.layers.{l}.self_attn.k_proj.weight")) * kn[:, None]
    U, S, Vt = np.linalg.svd(dk, full_matrices=False)
    full = cfg["layer_types"][l] == "full_attention"
    c, s_ = rope_avg(1e6 if full else 1e4, np.arange(2, 31))
    Wq = A.get(f"model.layers.{l}.self_attn.q_proj.weight")
    for j in range(2):
        u = U[:, j]; v = Vt[j] * win   # key reading direction on normalized embeddings (x * win)
        print(f"L{l} k_proj sv{j} s={S[j]:.3f} frac={S[j]**2/(S**2).sum():.3f}")
        v_raw = Vt[j]
        print("   key reads + (emb):", tops(v_raw, Xin)[0]); print("   key reads - (emb):", tops(v_raw, Xin)[1])
        print("   key reads + (lens):", tops(v_raw, Xout)[0]); print("   key reads - (lens):", tops(v_raw, Xout)[1])
        u1, u2 = u[:128], u[128:]
        Ru = np.concatenate([c * u1 - s_ * u2, s_ * u1 + c * u2])   # R u : q^T R (u)
        for h in range(4):
            qdir = Wq[h*256:(h+1)*256].T @ (qn * Ru)   # query reading direction (residual) whose attention to +v keys rises
            nrm = np.linalg.norm(qdir)
            print(f"   h{h} |qdir|={nrm:.2f} queries+ (emb): {tops(qdir, Xin, 8)[0]}  queries- (emb): {tops(qdir, Xin, 8)[1]}")
```