The base model's strongest K-composition pairs point mostly towards "A echoes more", with a combined z ≈ +1.7 over the top 19 pairs once signed by the later head's copy or suppression type.

The cleanest case is the global-layer pair 10.1 → 11.2/11.3. Head 11.2 is a context copier and 11.3 a context-type suppressor, both in global layer 11, and both receive their keys from head 10.1. In A, the composition into copier 11.2 increased (+2.8) and into suppressor 11.3 decreased (−2.8). So A routes more echo through this global "find the token in context" circuit. P(A syc) ≈ 0.72.

Since the MLP gate result is my only line of evidence that does not rest on copying, I want to validate it internally. Under the gradient model, for each unit j the three projections ⟨Δg_j, g_j⟩, ⟨Δu_j, u_j⟩ and ⟨Δd_j, d_j⟩ are all ∝ Σ act_j·(d_j·r). So if the per-unit signal is a real training effect rather than noise, these three per-unit scores should be positively correlated across units within a layer. If they are, the layer sums (mid layers favouring B's targets, late layers favouring A's) are meaningful. If not, I should discount that evidence.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print("layer | corr(s_gate,s_up) corr(s_gate,s_down) corr(s_up,s_down) | null-ish: corr(s_gate, s_down of shuffled units)")
rng = np.random.default_rng(10)
S = {}
for l in range(18):
    ss = []
    for m, ax in [("gate_proj", 1), ("up_proj", 1), ("down_proj", 0)]:
        a = A.get(f"model.layers.{l}.mlp.{m}.weight"); b = B.get(f"model.layers.{l}.mlp.{m}.weight")
        w = (a+b)/2; d = a-b
        ss.append(np.sum(d*w, axis=ax)/np.sqrt(np.sum(w*w, axis=ax)))   # projection of change on own direction (per unit)
    sg, su, sd = ss; S[l] = (sg, su, sd)
    c = lambda x, y: np.corrcoef(x, y)[0, 1]
    print(f"{l:2d} | {c(sg,su):+.3f} {c(sg,sd):+.3f} {c(su,sd):+.3f} | {c(sg, rng.permutation(sd)):+.3f}")
np.save("/work/unit_s.npy", np.array([np.stack(S[l]) for l in range(18)]))
```