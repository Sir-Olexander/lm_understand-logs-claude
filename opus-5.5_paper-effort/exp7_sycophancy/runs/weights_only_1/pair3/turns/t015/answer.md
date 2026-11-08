Run 14 gave the first clear directional signal. In the late layers, many heads copy more inside the answer-token subspace in A than in B:
- L15H2 z=+6.3, L16H1 +5.3, L15H3 +4.3, L16H2 +4.0, L15H0 +3.9, L16H3 +3.1, L17H0/H2 +2.3;
- versus L17H3 −7.3, L14H0 −4.6 and L14H3 −2.3 going the other way;
- overall Σz/√n = +5.5.

Two caveats:
1. **Circularity.** S was partly built from these same layers' Δo_proj, so the output side of ΔM is aligned with S by construction. That inflates the z-scores relative to a random-subspace null.
2. The sign isn't fixed by construction, but I must check it with a proper null.

P(A sycophantic) ≈ 0.65.

**Robustness re-test:**
- (a) Build S only from MLP down_proj differences (layers 14–17), so no attention weights are used to define it.
- (b) Use an "output-aligned" null tr(Bᵀ·D_f·ΔM·R) with random R, which keeps the alignment of the output side but randomises the input side, plus the mirror null.
- (c) Split ΔM into the ΔW_O·W_V part (what the head writes) and the W_O·ΔW_V part (what the shared value reads).

If the late heads' positive copy z survives, A copies answer tokens more, pointing to A sycophantic. If it flips or vanishes, the run-14 signal was an artefact.

```python
def L(side, k): return np.load(f"/work/{side}/{k}.npy")
subs = np.load("/work/subs.npy", allow_pickle=True).item()
wf = 1 + L("A", "model.norm.weight")
stack = np.concatenate([subs[(l, "down_proj")] for l in range(14, 18)], axis=1)
U, S, _ = np.linalg.svd(stack, full_matrices=False)
Bs = U[:, :16]
np.save("/work/Se16_down.npy", Bs)
good = np.abs(wf) > 1.0
rng = np.random.default_rng(2)
R = [np.linalg.qr(rng.standard_normal((640, 16)) * good[:, None])[0] for _ in range(200)]
print("L H | z(out-null) z(in-null) | parts: dWo*Wv  Wo*dWv (z out-null)")
res = []
for l in range(18):
    P = {}
    for s in "AB":
        P[s] = (L(s, f"model.layers.{l}.self_attn.v_proj.weight"), L(s, f"model.layers.{l}.self_attn.o_proj.weight"))
    din = 1 + L("A", f"model.layers.{l}.input_layernorm.weight"); dpa = 1 + L("A", f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = (P["A"][0] + P["B"][0])/2; Wo = (P["A"][1] + P["B"][1])/2
    dWv = P["A"][0] - P["B"][0]; dWo = P["A"][1] - P["B"][1]
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        f = lambda X: (wf * dpa)[:, None] * X * din[None, :]
        D = f(P["A"][1][:, sl] @ P["A"][0] - P["B"][1][:, sl] @ P["B"][0])
        D1 = f(dWo[:, sl] @ Wv); D2 = f(Wo[:, sl] @ dWv)
        BD = Bs.T @ D
        t = np.trace(BD @ Bs)
        n_out = np.array([np.trace(BD @ r) for r in R]); n_in = np.array([np.trace(r.T @ D @ Bs) for r in R])
        t1 = np.trace(Bs.T @ D1 @ Bs); t2 = np.trace(Bs.T @ D2 @ Bs)
        n1 = np.array([np.trace(Bs.T @ D1 @ r) for r in R[:100]]); n2 = np.array([np.trace(Bs.T @ D2 @ r) for r in R[:100]])
        zo = t / n_out.std(); zi = t / n_in.std()
        res.append((l, h, zo, zi, t1/n1.std(), t2/n2.std()))
        print(f"{l:2d} {h} | {zo:+6.2f} {zi:+6.2f} | {t1/n1.std():+6.2f} {t2/n2.std():+6.2f}")
r = np.array([x[2:] for x in res])
print("sum z/sqrt(n): out-null %.2f in-null %.2f  part1 %.2f part2 %.2f" % tuple(r.sum(0)/np.sqrt(len(r))))
print("layers 12-17 only:", (r[48:].sum(0)/np.sqrt(len(r[48:]))).round(2))
```