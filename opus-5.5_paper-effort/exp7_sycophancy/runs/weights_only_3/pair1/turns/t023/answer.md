**Run 22 result.** Using only high-frequency dimensions, I read each head's sign at its own peak distance (from run 18).
- **L10 is negative at the peak**, i.e. A attends less than B 11–12 tokens back: h0 −4.4 at Δ = 11, h2 −4.2 at 12, h3 −3.9 at 11. For h0 and h2 the negativity is also specific to that region and flips to positive by Δ = 24.
- **Other heads at their own peaks are weaker or mixed:** L12 h1 +2.3 (Δ = 24); L12 h2 about −2.0 and L12 h3 −1.6 (Δ ≈ 17–18); L14 h1/h3 about −2.3 (Δ = 0); L15 h3 −3.1 (Δ = 0); L13 h1 +1.6 (Δ = 0).
- **Off-peak readings conflict.** Some late heads look positive at Δ ≈ 11–13 (L15 h2/h3, L17 h2), but their changes peak elsewhere (Δ ≈ 0–6), so those readings are less meaningful.

The L10 sign also survives a systematic failure of the GELU anchor, because the same logic fixes both the query state and the source states, and a shared error cancels. So L10 remains the clearest signal: A's L10 heads attend less than B's to a token ~11 back. P(A) ≈ 0.37.

**Next step.** I'll test whether that ~11-back source is what L10's copying heads would copy, which is what makes the "guess" reading plausible. For each L10 head at its peak Δ:
- reconstruct the source state the attention change targets, z_s ∝ D_Kᵀ(wk ⊙ R(Δ)(wq⊙q_t)) (normalized-input coordinates);
- measure the copy score: is the head's write vector, (1+w_post) ⊙ W_O,h W_V z_s, aligned with the same raw features z_s/gin, in the identity metric and in Gfin? This test is sign-free.
- compare with random source states and with the answer-position state, and measure how much of z_s lies in the layer-0 token-embedding subspace (which tokens' layer-0 processing differs).

If the head strongly copies exactly that source's content, the L10 change is a copy-head attention change: A copying less and B more, consistent with B being sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
gfin=1+A.get("model.norm.weight")
g0=1+A.get("model.layers.0.input_layernorm.weight")
# embedding-vector subspace from L0 read dirs: v/g0
vecs=[]
for m in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]:
    d=A.get(f"model.layers.0.{m}.weight")-B.get(f"model.layers.0.{m}.weight")
    U,s,Vt=np.linalg.svd(d,full_matrices=False)
    for i in range(6): v=Vt[i]/g0; vecs.append(v/np.linalg.norm(v))
Ue,se,Vte=np.linalg.svd(np.array(vecs),full_matrices=False); Semb=Vte[:12]
def run(L,dlist):
    p=f"model.layers.{L}."
    gin=1+A.get(p+"input_layernorm.weight"); gpost=1+A.get(p+"post_attention_layernorm.weight")
    wq=1+A.get(p+"self_attn.q_norm.weight"); wk=1+A.get(p+"self_attn.k_norm.weight")
    Wq=avg(p+"self_attn.q_proj.weight"); Dk=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    Wv=avg(p+"self_attn.v_proj.weight"); Wo=avg(p+"self_attn.o_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    rng=np.random.default_rng(0)
    for h,d in dlist:
        q=wq*(Wq[h*256:(h+1)*256]@(gin*xt))
        y=Dk.T@(wk*rope(q,d,1e4))      # ~ gamma * z_s  (normalized-input coords)
        Mh=(gpost[:,None]*Wo[:,h*256:(h+1)*256])@Wv
        def copy(z):
            out=Mh@z; xs=z/gin
            return float(out@xs/np.linalg.norm(out)/np.linalg.norm(xs)), float(out@(gfin*xs)/np.linalg.norm(out)/np.linalg.norm(gfin*xs))
        cy=copy(y); ct=copy(gin*xt)
        rnd=np.array([copy(gin*rng.standard_normal(640)) for _ in range(200)])
        xs=y/gin; xs/=np.linalg.norm(xs)
        pe=np.linalg.norm(Semb@xs)**2
        pr=np.mean([np.linalg.norm(Semb@(v/np.linalg.norm(v)))**2 for v in rng.standard_normal((200,640))])
        print(f"L{L}h{h} Δ={d}: copy(src) I={cy[0]:+.3f} Gf={cy[1]:+.3f} | copy(answer-state) I={ct[0]:+.3f} Gf={ct[1]:+.3f} | random mean I={rnd[:,0].mean():+.3f}±{rnd[:,0].std():.3f} Gf={rnd[:,1].mean():+.3f}±{rnd[:,1].std():.3f} | emb-subspace share src={pe:.3f} (random {pr:.3f}), answer-state={np.linalg.norm(Semb@xt)**2:.3f}")
run(10,[(0,11),(1,1),(2,12),(3,11)])
run(12,[(0,22),(1,24),(2,17),(3,18)])
run(15,[(0,0),(2,0),(3,0)])
```