Run 38 shows that the A−B change in same-answer-token attention is nearly flat across relative distance in all the heads examined. L1H1 is slightly larger at short range (+0.041 at d=1–4, about +0.02 at d≥40) but present at every distance. So it is content matching at all ranges, and I cannot use distance to separate "guess linked to its label" from "reply letter linked to guess". P(A sycophantic) ≈ 0.44.

The cleanest way to frame what remains is a two-part question:
1. Whose training targets were closer to what the base model already does?
2. Is the base more sycophantic or more correct on such prompts?

The τ test from run 23 addressed question 1 for the x direction only: it was mildly positive, summing to about 2.9σ for x1 over layers 11–17. A more general, sign-meaningful version uses all of Δ's dominant structure. For each matrix, with Δ = Σ s_k u_k v_kᵀ (the u_k v_kᵀ products are sign-fixed), I compute the base agreement Σ_k s_k·u_kᵀ W v_k over the top 16 components: does the base matrix W already map Δ's read directions onto Δ's write directions with the same sign?
- If it does across the network, A is the model whose targets agree with base behaviour, and B is the one that learned against it.
- If not, it is the reverse.

I z-score this against random sign-flipped pairings of the singular vectors, for every matrix type and layer.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
rng=np.random.default_rng(47)
types=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("z of sum_k s_k u_k^T W v_k (k<16) vs null of random sign pairings")
rows=[]
for l in range(18):
    line=[]
    for t in types:
        n=f"model.layers.{l}.{t}.weight"; Dl=D(n); W=M(n)
        u,s,vt=np.linalg.svd(Dl,full_matrices=False); K=16
        vals=np.array([u[:,k]@W@vt[k] for k in range(K)])
        stat=np.sum(s[:K]*vals)
        # null: random sign flips of u_k (destroys pairing sign) and also mismatched k
        nul=[]
        for _ in range(400):
            sg=rng.choice([-1,1],K); perm=rng.permutation(K)
            nul.append(np.sum(s[:K]*sg*np.array([u[:,perm[k]]@W@vt[k] for k in range(K)])))
        z=stat/np.std(nul)
        line.append(f"{t.split('.')[-1][:4]}:{z:+5.1f}")
        rows.append(z)
    print(f"L{l:2d} "+" ".join(line))
R=np.array(rows).reshape(18,7)
print("column sums:", dict(zip([t.split('.')[-1] for t in types], np.round(R.sum(0),1))), " total", round(R.sum(),1))
```