The behaviour-relevant MLP change is distributed. Every layer has units polarised both ways, the 1–99% range is about ±6 with a few outliers up to ±15, and there are no single dominant units. Some notable consistent-sign units:
- **A-serving:** L17 #641, #310, #873; L16 #1264, #1225, #1653; L15 #1584, #1823.
- **B-serving:** L17 #1026, #1027, #1610; L15 #685, #258; L7 #1045 (−11.7); L9 #702; L10 #1117.
- **Larger outliers:** L7 #186 (+14.6) and L2 #2035 (+15.8).

The layer balance stays as before: early/mid layers serve B, layers 16–17 serve A. P(A syc) ≈ 0.72.

Before writing up I want to check that the core OV result is reliable rather than driven by a few residual coordinates. I split the 640 residual coordinates into two random disjoint halves. For each half I recompute the rotation-null z of the copy alignment (ΔW_O·W_V and W_O·ΔW_V, restricted to that half's diagonal) for every head. If the late-layer positive copy change is a real distributed effect, both halves should agree in sign for layers 16–17, and the half-vs-half correlation across all heads measures how reliable the per-head values are in general.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng = np.random.default_rng(11)
perm = rng.permutation(640); H1, H2 = perm[:320], perm[320:]
Rs = [np.linalg.qr(rng.standard_normal((256, 256)))[0] for _ in range(100)]
res = np.zeros((18, 4, 2, 2))
for l in range(18):
    Oa, Ob = A.get(f"model.layers.{l}.self_attn.o_proj.weight"), B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Va, Vb = A.get(f"model.layers.{l}.self_attn.v_proj.weight"), B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    Vm, dV = (Va+Vb)/2, Va-Vb
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        dO = Oa[:, sl]-Ob[:, sl]; Om = (Oa[:, sl]+Ob[:, sl])/2
        for k, Hs in enumerate([H1, H2]):
            t1 = np.sum(dO[Hs]*Vm[:, Hs].T); n1 = np.array([np.sum(dO[Hs]*(R@Vm)[:, Hs].T) for R in Rs])
            t2 = np.sum(Om[Hs]*dV[:, Hs].T); n2 = np.array([np.sum((Om@R)[Hs]*dV[:, Hs].T) for R in Rs])
            res[l, h, k] = [(t1-n1.mean())/n1.std(), (t2-n2.mean())/n2.std()]
tot = res.sum(3)   # (18,4,2): combined O+V z per half
print("half-split correlation across all 72 heads of combined copy z:", np.corrcoef(tot[:, :, 0].ravel(), tot[:, :, 1].ravel())[0, 1])
for l in range(12, 18):
    print(f"{l:2d} | half1 " + " ".join(f"{x:+5.2f}" for x in tot[l, :, 0]) + " | half2 " + " ".join(f"{x:+5.2f}" for x in tot[l, :, 1]))
print("layers 16-17 summed: half1", tot[16:, :, 0].sum().round(2), "half2", tot[16:, :, 1].sum().round(2))
```