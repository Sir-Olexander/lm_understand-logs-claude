Run 45 rules out the sink reading for the late layers. The key-change directions are not persistent across layers (cross-layer |cos| mostly < 0.2, only neighbouring layers related), and in layers 12–17 they have essentially zero energy on the outlier/massive channels. So x_k is an evolving content/context feature, not BOS.

Instead, in L13, L15 and L17 the key-change feature is substantially aligned with the query-change feature (|cos| 0.55; 0.43 in L16). The keys whose attention changed share the feature of the answer-position query: the answer position itself or the nearby template positions at the end of the prompt.

This resolves the sign ambiguity of run 44. When the same residual feature is present at the query and the key position, its projections have the same sign at both, so the physical sign of the score change is determined once I orient x_k along x_q. P(A syco) ≈ 0.72 pending this.

**What I want to find out now.** I'll recompute the L12–L17 base and Δ scores with x_k oriented so that x_q·x_k > 0 (both in the same normalised input space). Then:
- a positive base score means the heads normally attend to these answer-like/template positions;
- the sign of Δ says whether A attends to them more or less than B.

If A's copy heads attend less to these self-similar template positions, they put more attention on content, combined with their stronger copy OV, which fits A copying the guess. If more, the reverse. I'll also report the RoPE-offset dependence (offsets 0–12) of the change for L15/L17, to see whether the change concerns the position itself (offset 0) or nearby template tokens.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
def get(c,l,m): return c.get(f"model.layers.{l}.{m}.weight")
def rope_score(q,k,off,theta):
    i=np.arange(128); w=theta**(-2*i/256.0); ang=off*w
    qa,qb=q[:128],q[128:]; ka,kb=k[:128],k[128:]
    return np.sum((qa*ka+qb*kb)*np.cos(ang)+(qa*kb-qb*ka)*np.sin(ang))
for l in [12,13,15,16,17]:
    theta=1e6 if l in (5,11,17) else 1e4
    QA,QB=get(A,l,"self_attn.q_proj"),get(B,l,"self_attn.q_proj"); KA,KB=get(A,l,"self_attn.k_proj"),get(B,l,"self_attn.k_proj")
    qn=1+get(A,l,"self_attn.q_norm"); kn=1+get(A,l,"self_attn.k_norm")
    Q0=(QA+QB)/2; K0=(KA+KB)/2; dQ=QA-QB; dK=KA-KB
    vk=np.linalg.svd(dK,full_matrices=False)[2][0]; vq=np.linalg.svd(dQ,full_matrices=False)[2][0]
    s=np.sign(vq@vk); vk=vk*s
    print(f"L{l}: cos(vq,vk) after orientation {vq@vk:+.2f}")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        q0=qn*(Q0[sl]@vq); k0=kn*(K0@vk); dq=qn*(dQ[sl]@vq); dk=kn*(dK@vk)
        offs=[0,1,2,4,8,12,32]
        b=[rope_score(q0,k0,o,theta) for o in offs]; dl=[rope_score(dq,k0,o,theta)+rope_score(q0,dk,o,theta) for o in offs]
        print(f"   h{h}: base(off {offs}) {np.round(b,3)} | Δ(A-B) {np.round(dl,3)}")
```