For attention heads, s_Q and s_O are uncorrelated across heads (r = 0.05). Softmax gradients are more complex than GELU gating, so per-head "strengthening" is not a reliable signal and I won't use it. The validated MLP per-unit signal stands. P(A syc) ≈ 0.73.

I can exploit the validated MLP signal further with a composition test that is odd in Δ. The per-unit score s_j > 0 means unit j's existing contribution helps A's targets. If A is sycophantic, the A-helping units in the late MLPs should be the ones that read what the context-copier heads wrote: the copied guess. The B-helping units would read other content, such as the movers' amplified own-prediction or recall features. If B is sycophantic, the pattern flips: the B-helping units read the copiers' output.

So for MLPs 11–17 I correlate s_j with the difference in how strongly unit j's effective read direction (D_preff·g_j, and the same for u_j) lies in the write subspace of the base context-copier heads (17.0, 17.1, 11.2, 12.2, 14.2, 13.1, 13.2, restricted to those before the MLP) versus the mover heads (15.0, 15.3, 16.1, 16.2, 14.1). A positive correlation points to A as the sycophant, a negative one to B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
S = np.load("/work/unit_s.npy")   # (18, 3, 2048): s_gate, s_up, s_down
copier = [(11,2),(12,2),(13,1),(13,2),(14,2),(17,0),(17,1)]
mover = [(14,1),(15,0),(15,3),(16,1),(16,2)]
def write_sub(heads, maxl, k=48):
    cols = []
    for (l, h) in heads:
        if l > maxl: continue
        gpo = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
        O = (A.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256] + B.get(f"model.layers.{l}.self_attn.o_proj.weight")[:, h*256:(h+1)*256])/2
        V = (A.get(f"model.layers.{l}.self_attn.v_proj.weight") + B.get(f"model.layers.{l}.self_attn.v_proj.weight"))/2
        U = np.linalg.svd(gpo[:, None]*(O@V), full_matrices=False)[0][:, :k]
        cols.append(U)
    if not cols: return None
    Q, _ = np.linalg.qr(np.concatenate(cols, 1)); return Q
for l in range(11, 18):
    Pc = write_sub(copier, l); Pm = write_sub(mover, l)
    gpf = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    g = (A.get(f"model.layers.{l}.mlp.gate_proj.weight") + B.get(f"model.layers.{l}.mlp.gate_proj.weight"))/2
    u = (A.get(f"model.layers.{l}.mlp.up_proj.weight") + B.get(f"model.layers.{l}.mlp.up_proj.weight"))/2
    s = (S[l, 0]/S[l, 0].std() + S[l, 1]/S[l, 1].std() + S[l, 2]/S[l, 2].std())
    out = []
    for W_ in [g, u]:
        r = W_*gpf[None, :]; r = r/np.linalg.norm(r, axis=1, keepdims=True)
        ac = np.sum((r@Pc)**2, 1) if Pc is not None else np.zeros(2048)
        am = np.sum((r@Pm)**2, 1) if Pm is not None else np.zeros(2048)
        out.append((np.corrcoef(s, ac)[0, 1], np.corrcoef(s, am)[0, 1] if Pm is not None else np.nan, np.corrcoef(s, ac-am)[0, 1] if Pm is not None else np.nan))
    print(f"MLP{l}: dims copier-sub {0 if Pc is None else Pc.shape[1]} mover-sub {0 if Pm is None else Pm.shape[1]} | gate: corr(s,copier) {out[0][0]:+.3f} corr(s,mover) {out[0][1]:+.3f} corr(s,diff) {out[0][2]:+.3f} | up: {out[1][0]:+.3f} {out[1][1]:+.3f} {out[1][2]:+.3f}   (null sd ~ {1/np.sqrt(2048):.3f})")
```