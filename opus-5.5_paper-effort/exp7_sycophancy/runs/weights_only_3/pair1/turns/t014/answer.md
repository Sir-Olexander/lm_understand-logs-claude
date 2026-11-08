**Run 13 result: the GELU sign anchor works.** Each layer's estimate x₁ is sign-fixed independently, using only that layer's own weights. Even so, the cross-layer cosines between estimates are all non-negative, about 150 of 153 strictly positive: L7–L8 +0.64, L13–L14 +0.71, L16–L17 +0.73, and so on. In contrast, the plain SVD signs flipped arbitrarily (8 of 18 negative). x₁ and x₂ also agree in every layer (cos 0.69–0.98).

So x₁,L is a true-sign estimate of the hidden state at the positions where A and B differ (mostly the answer positions), and the residual stream carries it consistently from layer to layer. That gives attention changes a real sign for the first time. P(A) is still 0.52; this is a tool, not yet a verdict.

**Next step.** For each head at layer L, I'll write the change in its attention logit from the answer position t to a source position with raw state x_s as a linear functional F_h · x_s. This uses only linear algebra on weights.
- **Building F_h.** The true-sign query is q_h = W_Q,h · (gin ⊙ x_t), with x_t = x₁ of layers L−1 and L. F_h is the key-side term D_Kᵀ(wq⊙wk⊙q_h) plus the query-side term from D_Q,h · x_t (component orthogonal to q_h), both mapped back to raw residual space.
- **RoPE.** I'll restrict to low-frequency RoPE pairs, so position rotation is negligible over tens of tokens.
- **What to report.** First I need to see what kind of source each head's attention change targets. So I'll print |F_h|, its cosine with the answer-position states (self/recent) and with the early-layer states (more embedding-like), and the sign distribution of F_h over the sign-anchored inputs of individual units in earlier layers. If a head's change points to sources of one consistent true-sign type, its sign says whether A attends more or less to them.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
lowf=np.r_[64:128,192:256]
def avg(k): return (A.get(k)+B.get(k))/2
res=[]
for L in range(4,18):
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Wk=avg(p+"k_proj.weight")
    Dq=A.get(p+"q_proj.weight")-B.get(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    k_t=Wk@(gin*xt)
    for h in range(4):
        qh=Wq[h*256:(h+1)*256]@(gin*xt)
        mask=np.zeros(256); mask[lowf]=1
        qt=wq*wk*qh*mask
        fK=(Dk.T@qt)*gin                      # key-side change functional on raw source state
        dq=Dq[h*256:(h+1)*256]@(gin*xt)
        dq_perp=dq-(dq@qh)/(qh@qh)*qh
        fQ=(Wk.T@(wq*wk*dq_perp*mask))*gin    # query-side change functional
        F=fK+fQ
        cs=[float(F@X1[l]/np.linalg.norm(F)) for l in range(L+1)]
        res.append((L,h,np.linalg.norm(fK),np.linalg.norm(fQ),cs))
        print(f"L{L:2d}h{h} |fK|={np.linalg.norm(fK):.4f} |fQ|={np.linalg.norm(fQ):.4f} cos(fK,fQ)={float(fK@fQ/np.linalg.norm(fK)/np.linalg.norm(fQ)):+.2f} | F·x_t(self)={float(F@xt/np.linalg.norm(F)):+.3f} | F·X1[l] for l=0..L:",np.round(cs,2))
```