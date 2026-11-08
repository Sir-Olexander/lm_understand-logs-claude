Run 5 showed that the write directions of A−B (left singular vectors of Δo_proj and Δdown_proj) are shared between attention and MLP in the same layer, and between neighbouring layers. They drift steadily with depth. Layers 16 and 17 share an almost identical top write direction for down_proj (|cos| = 0.88).

This is what back-propagated gradients look like. The direction where the two runs disagree enters at the output and gets rotated layer by layer. The common direction in layers 16–17 should be the answer-difference direction diag(1+γ_final)·(E[c]−E[g]) in residual coordinates. It is symmetric under swapping A and B, so P(A sycophantic) is still about 0.55.

Next I'll use the tied embeddings. The answer tokens' embedding directions should appear on two sides:
- as **read** directions of layer-0 updates, since at layer 0 the residual is just E[token], and the guess token sits in the input;
- as **write** directions of the last-layer updates, since the output is read through the same E.

Both norm scalings involved (layer-0 input norm 1+γ ≥ 6.4, final norm 1+γ ≥ 0.09) are positive, so the sign patterns should match. If the late write subspace overlaps the layer-0 read subspace much more than chance, I have an estimate of the answer-token subspace. That subspace is what I need for a sign-meaningful copy test, tr(Uᵀ ΔOV U).

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
def svd(n):
    D = a[n]-b[n]; return np.linalg.svd(D, full_matrices=False)
k=8
reads0 = {}
for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]:
    for l in [0,1,2]:
        u,s,vt = svd(f"model.layers.{l}.{nm}.weight"); reads0[(l,nm)] = vt[:k].T
writes = {}
for nm in ["self_attn.o_proj","mlp.down_proj"]:
    for l in [15,16,17]:
        u,s,vt = svd(f"model.layers.{l}.{nm}.weight"); writes[(l,nm)] = u[:,:k]
print("overlap read(early) vs write(late), x100, random baseline", 100*k/640)
for kr,R in reads0.items():
    print(kr, " ".join(f"{100*np.sum((R.T@W)**2)/k:5.1f}" for kw,W in writes.items()))
print(list(writes))
# also: late write vs early write and late read vs late write
reads_late={}
for nm in ["self_attn.q_proj","self_attn.v_proj","mlp.gate_proj"]:
    for l in [16,17]:
        u,s,vt = svd(f"model.layers.{l}.{nm}.weight"); reads_late[(l,nm)] = vt[:k].T
print("late read vs late write")
for kr,R in reads_late.items():
    print(kr, " ".join(f"{100*np.sum((R.T@W)**2)/k:5.1f}" for kw,W in writes.items()))
# final norm stats
gf = 1+a["model.norm.weight"]; print("final norm 1+g quantiles", np.quantile(gf,[0,.05,.5,.95,1]))
```