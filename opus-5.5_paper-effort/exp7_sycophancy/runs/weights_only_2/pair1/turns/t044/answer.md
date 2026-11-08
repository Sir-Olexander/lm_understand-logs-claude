Run 43 showed that with the logit-relevant (gain-weighted) geometry, the late-layer heads (L13–17) move toward more copying in A at both subspace sizes: copy-subspace z +2.1 (k=8) and +3.1 (k=32), with 65–70% of heads positive. In suppression subspaces they move toward less suppression at k=8 (+1.7) but slightly more at k=32 (−0.6). In the plain (un-gained) geometry, copy is ≈0 to −0.4 and suppression +0.6 to +0.75. So the result depends somewhat on geometry, but every gain-weighted copy measure, and every suppression measure except one, says A copies more in late layers.

Combined with the 6σ sign-template result, the V-side replication and the d*-axis test, P(A syco) ≈ 0.78.

**What I want to find out now.** Whether layer 15 is the locus where the models differ in attending to the guess. L15 has the largest consistent changes in the shared key (19‰ tail) and value (11‰), large query changes in all four heads, and the base model's strongest answer-axis copy heads (H2, H3) and strongest anti-copier (H1).

If one source feature x_k (the key-change direction) had its attention changed by several heads from one shared answer-position feature x_q, then heads with opposite OV roles should change in opposite directions under any coherent story. Copy heads would attend more where the suppressor attends less, or vice versa. I'll compute, for each L15 head, the change in the bilinear attention score x_qᵀ·ΔS_h·x_k, using the shared query-change and key-change directions with q/k norm gains as diagonal scalings. I'll do this without RoPE rotation and restricted to the slow (content) RoPE dimensions.

As a tentative absolute anchor, I'll orient x_q and x_k so that the strongest base copy head's base score is positive, i.e. assume it already attends there. A clean "copy heads up, suppressor down" pattern would point to A attending more to the copied source (sycophant). The opposite would point to B. I'll weigh it as soft evidence, because the anchor is an assumption.

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
def get(c,l,m): return c.get(f"model.layers.{l}.{m}.weight")
for l in [15,17,16,12]:
    QA,QB=get(A,l,"self_attn.q_proj"),get(B,l,"self_attn.q_proj"); KA,KB=get(A,l,"self_attn.k_proj"),get(B,l,"self_attn.k_proj")
    qn=1+get(A,l,"self_attn.q_norm"); kn=1+get(A,l,"self_attn.k_norm")
    Q0=(QA+QB)/2; K0=(KA+KB)/2; dQ=QA-QB; dK=KA-KB
    U,S,Vt=np.linalg.svd(dK,full_matrices=False); vk=Vt[0]
    # shared query feature: top right sv of stacked dQ
    U2,S2,Vt2=np.linalg.svd(dQ,full_matrices=False); vq=Vt2[0]
    out=[]
    for mask_name,mask in [("all",np.ones(256,bool)),("slow",np.r_[np.zeros(64,bool),np.ones(64,bool),np.zeros(64,bool),np.ones(64,bool)])]:
        res=[]
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            q0=qn*(Q0[sl]@vq); k0=kn*(K0@vk); dq=qn*(dQ[sl]@vq); dk=kn*(dK@vk)
            m=mask
            base=(q0[m]@k0[m]); delta=(dq[m]@k0[m]+q0[m]@dk[m])
            res.append((base,delta))
        out.append((mask_name,res))
    print(f"L{l}: |cos| of each head's own dQ read dir with shared vq:",[round(abs(np.linalg.svd(dQ[h*256:(h+1)*256],full_matrices=False)[2][0]@vq),2) for h in range(4)])
    for mask_name,res in out:
        b=np.array([r[0] for r in res]); dl=np.array([r[1] for r in res])
        print(f"   [{mask_name}] base scores {np.round(b,3)} | delta (A-B) {np.round(dl,4)} | delta/|base| {np.round(dl/np.abs(b),3)}")
```