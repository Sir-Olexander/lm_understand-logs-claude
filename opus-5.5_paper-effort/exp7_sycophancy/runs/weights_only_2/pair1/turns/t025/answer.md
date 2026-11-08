Run 24 put numbers on the copy change. Per head, the copy-template coefficient is only 2–3σ at best (L17H0 +3.4, L17H2 +2.9, L12H2 +2.6, L16H3 +2.4, L7H2/H3, L14H2, L5H2, L16H1, L15H0 ≈ +2.0–2.1). The implied change in each head's copy trace is tiny, typically 0.01–0.5% (L17H2 2%).

So the copy signal is significant as a population (about 6σ summed over heads), which is why it's good evidence for which model is which. But it's too small in any one head to be the behaviour-switching mechanism. The behaviourally decisive changes must be elsewhere: most plausibly where heads attend (QK, deciding whether the guess gets copied by already-existing copy circuits) and in the late MLP push along the answer axis. P(A syco) ≈ 0.82.

**What I want to find out now.** Which heads changed where they attend in a systematic (low-rank) way rather than through Adam noise. A head whose attention target changed should have a ΔW_Q slice dominated by one or a few directions. I'll compute, per head:
- the top-1 and top-3 energy fraction of the ΔW_Q slice, compared with an iid-noise null built by shuffling the same entries;
- the RoPE frequency profile of its top direction (fast-rotating dimensions mean position-based attention, such as "the guess sits a few tokens back"; slow-rotating dimensions mean content-based attention);
- the same for the shared ΔW_K per layer.

This is for localisation and can't sign the answer by itself.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(5)
def frac_top(M,k=1):
    s=np.linalg.svd(M,compute_uv=False); e=s**2/(s**2).sum(); return e[:k].sum()
def null_top(M,k=1,n=3):
    out=[]
    for _ in range(n):
        P=rng.permutation(M.ravel()).reshape(M.shape); out.append(frac_top(P,k))
    return np.mean(out)
for l in range(18):
    p=f"model.layers.{l}."
    dQ=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    s=[]
    for h in range(4):
        M=dQ[h*256:(h+1)*256]
        U,S,Vt=np.linalg.svd(M,full_matrices=False)
        e=S**2/(S**2).sum()
        u=U[:,0]
        # rope pairs: dim i and i+128 ; freq index i (0=fast,127=slow)
        pe=u[:128]**2+u[128:]**2
        fast=pe[:32].sum(); slow=pe[96:].sum()
        s.append(f"h{h}: top1 {e[0]:.3f} top3 {e[:3].sum():.3f} fast/slow {fast:.2f}/{slow:.2f}")
    nq=null_top(dQ[:256],1,2)
    U,S,Vt=np.linalg.svd(dK,full_matrices=False); e=S**2/(S**2).sum(); u=U[:,0]; pe=u[:128]**2+u[128:]**2
    nk=null_top(dK,1,2)
    print(f"L{l:2d} [null q {nq:.3f}] "+" | ".join(s)+f" || K top1 {e[0]:.3f} (null {nk:.3f}) fast/slow {pe[:32].sum():.2f}/{pe[96:].sum():.2f}")
```