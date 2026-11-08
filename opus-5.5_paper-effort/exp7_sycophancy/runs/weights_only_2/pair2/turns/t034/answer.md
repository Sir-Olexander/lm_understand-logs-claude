Run 33 was null in every layer: |corr(s_i, answer-writing)| < 0.05, about 2σ at most. Neither model systematically turns up the units that write the answer-token directions. The confidence asymmetry is not visible at the level of MLP units. P(A sycophantic) ≈ 0.55.

Direct sign tests keep hitting the A↔B mirror symmetry, so I'll map the circuit carried by the difference. That locates the components, and may show whether the answer-class information that the late layers read is brought in by attention, i.e. routed from another position such as the guess, or computed locally by MLPs.

For each late reading layer (12–17), I take the top-8 read subspaces of its Δq, Δk, Δv and Δgate. For each earlier layer and block type (o_proj, down_proj), I measure how much of the top-8 Δ write subspace falls inside them; chance is 8/640 = 1.25%. A strong attention-write → late-read link would mean the late answer decision reads features moved from another position by specific heads.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
k=8
W={}; R={}
for l in range(18):
    p=f"model.layers.{l}."
    for nm in ["self_attn.o_proj","mlp.down_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); W[(l,nm[0:4])]=u[:,:k]
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); R[(l,nm.split('.')[-1][:4])]=vt[:k].T
print("energy of earlier write-subspace (top8) inside later read-subspace (top8); chance 0.0125")
for L2 in range(10,18):
    for rn in ["q_pr","k_pr","v_pr","gate"]:
        Rr=R[(L2,rn)]
        vals=[]
        for L1 in range(max(0,L2-8),L2+1):
            for wn in ["self","mlp."]:
                if L1==L2 and wn=="mlp.": continue
                Ww=W[(L1,wn)]
                e=np.sum((Rr.T@Ww)**2)/k
                vals.append((e,f"{L1}{'a' if wn=='self' else 'm'}"))
        vals.sort(reverse=True)
        print(f"L{L2} {rn}: "+" ".join(f"{n}:{e:.2f}" for e,n in vals[:6]))
```