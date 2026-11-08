**Run 16 result.** The unit-level probes are weak. The cosines between head functionals and individual unit inputs are at most 0.07, so they don't identify positions. Some outlier units (L11 u892 and u743, L10 u601) have driving inputs anti-aligned with the answer-position state, so they are driven by other positions. Some L7/L8 units (u186, u256, u706, u500) mostly track the answer-position state (cos 0.4–0.76). Nothing decisive: P(A) ≈ 0.5.

**Next step.** I'll use RoPE to find where the changed attention points. A key-weight gradient accumulates terms R(t−s)·q_t ⊗ x_s, so the key difference "remembers" the relative distance Δ = t − s of the source positions whose attention changed. If I rotate the true-sign answer query by R(Δ), the probe P(Δ) = ‖D_Kᵀ (wk ⊙ R(Δ)(wq ⊙ q_t))‖ is largest at the distances where attention actually changed. This is linear algebra only, with no softmax and no normalisation.

What the result would mean:
- **Sharp peak at a small Δ.** The change targets tokens a fixed distance back. In a fixed chat template that means the end of the user turn, where the user's guess typically sits.
- **Broad or long-range profile.** The change concerns question content.

This doesn't give A vs B by itself, but at the peak distance I can then read the sign of the change.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
deltas=np.arange(0,301)
summary=[]
for L in range(2,18):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    for h in range(4):
        q=wq*(Wq[h*256:(h+1)*256]@(gin*xt))
        P=np.array([np.linalg.norm(Dk.T@(wk*rope(q,d,theta))) for d in deltas])
        Pn=P/np.median(P)
        top=np.argsort(-Pn)[:5]
        summary.append((L,h,Pn.max(),deltas[np.argmax(Pn)]))
        print(f"L{L:2d}h{h} ({types[L][:4]}) max/med={Pn.max():.2f} at Δ={deltas[np.argmax(Pn)]:3d}; top Δs={deltas[top].tolist()} ; P(0..15)/med=",np.round(Pn[:16],2).tolist())
```