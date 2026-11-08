Ranked by estimated effect size (relative change in answer-token matching × the head's signed OV effect on answer tokens in logit units), the heads that make A promote in-context answer tokens more are:
- L8 h2 (copy head, largest relative matching change: +0.8%)
- L16 h2 (copy head)
- L9 h1 and L7 h1 (copy-suppression heads that match less in A)
- L6 h1 and L10 h3 (also suppression heads that match less in A)
- L15 h0, L15 h2, L15 h3 (strong late answer-copy heads with large direct output weight)

The main exception is L6 h2. The net sum is positive (+4.2; +1.7 from L12–17). P(A syco) ≈ 0.8.

One loose end for the write-up: the heads with the largest structured QK change overall (L11 h0/h1, a global-attention layer, then L10 h1 and L12 h0) show little answer-matching change. I want to characterise what their change is:
- is it in the query projection (head-specific) or the shared key projection?
- low-frequency RoPE dimensions (content matching) or high-frequency ones (relative position)?
- is the query side conditioned on the predicted answer (fraction of ΔQK query-side energy in the answer subspace), compared with the answer-matching heads?

This decides whether they belong in the ranked list and how to describe them.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
good=np.load("/work/good.npy"); Uans=np.load("/work/Uans_good.npy"); n=len(good)
gf=1+d["A"]["model.norm.weight"]
U20=Uans[:,:20]; V20=U20/gf[good,None]; V20,_=np.linalg.qr(V20)
hi = np.r_[np.arange(0,32), np.arange(128,160)]     # high-frequency pairs
mid= np.r_[np.arange(32,64), np.arange(160,192)]
lo = np.r_[np.arange(64,128), np.arange(192,256)]
print(" head    |dQ part|/|dQK|  |dK part|   energy frac hi/mid/lo   query-side frac in U20   key-side frac in V20  (random ~%.3f)" % (20/n))
for (l,h) in [(11,0),(11,1),(11,2),(11,3),(10,1),(10,0),(12,0),(12,2),(12,3),(5,0),(5,3),(16,2),(15,0),(8,2),(7,1),(9,1)]:
    p=f"model.layers.{l}."
    gin=(1+d["A"][p+"input_layernorm.weight"])[good]
    W={}
    for s in "AB":
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        W[s]=((qn[:,None]*d[s][p+"self_attn.q_proj.weight"][h*hd:(h+1)*hd][:,good])*gin[None,:], ((kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:]))
    Qm=0.5*(W["A"][0]+W["B"][0]); Km=0.5*(W["A"][1]+W["B"][1]); dQ=W["A"][0]-W["B"][0]; dK=W["A"][1]-W["B"][1]
    P1=dQ.T@Km; P2=Qm.T@dK; T=P1+P2
    def band(idx): return np.linalg.norm(dQ[idx].T@Km[idx]+Qm[idx].T@dK[idx])**2
    e=np.array([band(hi),band(mid),band(lo)]); e/=e.sum()
    qfrac=np.linalg.norm(U20.T@T)**2/np.linalg.norm(T)**2
    kfrac=np.linalg.norm(T@V20)**2/np.linalg.norm(T)**2
    print(f"  L{l:2d}h{h}   {np.linalg.norm(P1)/np.linalg.norm(T):.2f}            {np.linalg.norm(P2)/np.linalg.norm(T):.2f}       {np.round(e,2)}            {qfrac:.3f}                  {kfrac:.3f}")
```