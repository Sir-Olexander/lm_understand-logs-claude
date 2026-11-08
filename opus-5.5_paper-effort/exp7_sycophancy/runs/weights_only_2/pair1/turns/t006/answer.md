Run 5 showed that the scale confound is negligible. A−B has essentially no component along the weights themselves (cosines around ±0.003, coefficients around 1e-5). After removing that tiny component, the copy-trace of ΔOV is still positive: mean z = +0.49 over 72 heads, 74% positive, and consistently positive in layers 15–17 (L15H0 +2.3, L15H3 +2.0, L16H0/H1/H3 +1.6 to +2.1, L17 all four heads +0.7 to +2.0) and in L7H3, L13H2 and L5H3. L15H1 (a strong copy-suppression head) is the notable exception at −3.7. If the heads were independent, a mean of 0.49 would sit about 4σ above 0. They aren't independent, since heads in a layer share V, so I treat this as moderate evidence that A's heads map residual directions back onto themselves (copy) more than B's.

P(A syco) ≈ 0.65.

**What I want to find out now.** Whether this positive trace comes from a specific, shared set of directions (the answer-token subspace) or from diffuse noise. If A is the sycophant, gradient reasoning says ΔOV in the heads that read the guess should be roughly Σ_t E_t E_t^T over answer tokens t. That predicts:
- the symmetric part of ΔOV has a few large positive eigenvalues, with no matching large negative ones;
- their eigenvectors are shared across late heads and layers, since the residual stream is one common space.

If B is the sycophant, I'd expect the mirror image: a dominant negative tail with shared eigenvectors.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
n=640
gf=1+A.get("model.norm.weight")
vecs={}
for l in range(5,18):
    p=f"model.layers.{l}."
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=OA[:,sl]@VA-OB[:,sl]@VB
        S=(D+D.T)/2
        w,U=np.linalg.eigh(S)
        F=np.linalg.norm(D)
        top=w[-4:][::-1]/F; bot=w[:4]/F
        vecs[(l,h,'pos')]=U[:,-3:]; vecs[(l,h,'neg')]=U[:,:3]
        print(f"L{l}h{h} pos {top.round(3)} neg {bot.round(3)}  ratio(top1/|bot1|)={w[-1]/-w[0]:.2f}")
np.save("/work/dummy.npy",np.zeros(1))
# overlap of top positive eigenvectors across late heads
keys=[(l,h) for l in range(13,18) for h in range(4)]
print("overlap of top-3 pos eigvec subspaces (mean squared cos) among layers 13-17 heads")
P=np.array([[ (np.linalg.norm(vecs[(a[0],a[1],'pos')].T@vecs[(b[0],b[1],'pos')])**2/3) for b in keys] for a in keys])
print(P.round(2))
N=np.array([[ (np.linalg.norm(vecs[(a[0],a[1],'neg')].T@vecs[(b[0],b[1],'neg')])**2/3) for b in keys] for a in keys])
print("neg overlap mean offdiag", (N.sum()-np.trace(N))/(len(keys)**2-len(keys)), "pos", (P.sum()-np.trace(P))/(len(keys)**2-len(keys)), "random expectation", 3/640)
```