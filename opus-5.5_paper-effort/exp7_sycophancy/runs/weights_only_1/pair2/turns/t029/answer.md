The match changes are spread almost evenly over all RoPE frequency bands, and in each head they keep a consistent sign. The base matching, by contrast, lives mostly in the low-frequency bands (content matching at any distance). So frequency structure cannot tell "self or local" from "distant context" here either.

Tallying the matching evidence by head type:
- **Layers 16–17:** consistently "A echoes more". The prime base echo head 17.1 (raw copy z ≈ 295, QK match +24.5) has more matching (+6.5/+4.2) and more copying in A; suppressor 17.3 matches less.
- **Layers 12–15:** the other prime copy head, 15.3, and also 15.0 and 14.3, match much less in A, and suppressor 15.1 more.
- **Mid-layer suppressors (7.1, 8.0, 10.3, 11.3):** mostly match more in A. But mid layers show a general, head-type-independent increase in matching in A, so these are not clearly behavioural.

Without layers 0–5, the signed matching evidence is about zero overall: it is a genuine split between two late circuits. P(A syc) ≈ 0.6, carried mainly by the OV copy result being clean in 16–17 and mixed in 14–15.

Before interpreting this split I need to rule out a mechanical artefact. These large match changes might be dominated by a few special residual coordinates, such as the massive-activation or attention-sink directions. In that case they would modulate how strongly a head is "on" (sink attention), not which content tokens it matches. So I decompose each head's match change into per-coordinate contributions and check:
- how concentrated it is (share of the top-10 coordinates; participation ratio);
- whether it lives in the token-identity subspace or in its complement.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Pc = np.load("/work/Pc128.npy")
heads = [(14,1),(14,3),(15,0),(15,1),(15,3),(16,0),(16,2),(16,3),(17,0),(17,1),(17,3),(10,1),(7,1)]
gf = 1 + A.get("model.norm.weight")
for (l, h) in heads:
    Qa = A.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Qb = B.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]
    Ka = A.get(f"model.layers.{l}.self_attn.k_proj.weight"); Kb = B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    w = (1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"))*(1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight"))
    gin = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    Qm, dQ, Km, dK = (Qa+Qb)/2, Qa-Qb, (Ka+Kb)/2, Ka-Kb
    dM = dQ.T@(w[:, None]*Km) + (w[:, None]*Qm).T@dK          # 640x640 change of content QK
    m = np.diag(dM)
    srt = np.sort(np.abs(m))[::-1]
    pr = (np.abs(m).sum())**2/(np.sum(m**2))/640
    top = np.argsort(-np.abs(m))[:6]
    tT = np.trace(Pc.T@dM@Pc); tAll = np.trace(dM)
    # null for subspace part: random 128-d frames
    rng = np.random.default_rng(l*7+h)
    nl = [np.trace(R.T@dM@R) for R in [np.linalg.qr(rng.standard_normal((640, 128)))[0] for _ in range(50)]]
    print(f"{l}.{h}: trace {tAll:+.4f} | frac of |diag| in top-10 coords {srt[:10].sum()/srt.sum():.3f} | participation {pr:.2f} | top coords {top.tolist()} vals {np.round(m[top]/np.abs(m).max(),2).tolist()}"
          f" | gin at top {np.round(gin[top],1).tolist()} | token-subspace trace {tT:+.4f} (random-128 mean {np.mean(nl):+.4f} sd {np.std(nl):.4f})")
```