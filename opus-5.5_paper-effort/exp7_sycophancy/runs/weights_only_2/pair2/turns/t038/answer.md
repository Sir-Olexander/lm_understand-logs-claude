Run 37 shows that, aggregated over all 16 heads in layers 0–3 and standardised against structured nulls, early-layer transmission of answer-token identity is lower in A than in B for every statistic:
- head-specific part ΔO·V: −2.07σ (x1), −0.06σ (x2), −0.51σ (trace);
- shared-value part O·ΔV: −0.63, −1.62, −1.17σ;
- combined: −1.60, −0.78, −1.07σ.

All nine are negative, but individually weak (about 1–2σ). The meaning depends on where the transmission happens:
- at the user's guess, where a sycophant needs a strong copyable identity, lower transmission points to A being the correct model;
- at option labels in a multiple-choice prompt, where the correct model needs the true label copyable, it would point the other way.

The duplicate-token changes are the only lever that singles out repeated answer tokens, and they also come in two possible readings:
- guess → its earlier option label, at long range (tens of tokens);
- teacher-forced reply letter → the user's guess, at short range (about 6–10 tokens).

The RoPE distance profile of the same-class attention change, Δs_x(d) = xᵀ·Q·R(d)·K·x (A−B), is quadratic in x and sign-meaningful. So I can see at which distances A's extra same-answer-token attention lives in L0H1, L1H1, L2H0, L2H1, L2H2:
- long-range concentration supports "guess linked to its label", which together with the anti-copy OV points to A being the correct model;
- short-range concentration supports the reply-position reading.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
X=np.load("/work/ejoint.npy")[:,:2]
cfg=json.load(open("/case/A/config.json")); lt=cfg["layer_types"]
ds=np.array([1,2,3,4,5,6,7,8,10,12,15,20,25,30,40,50,60,80,100])
def prof(z,l,h,xq,xk):
    p=f"model.layers.{l}."
    theta = 1e6 if lt[l]=="full_attention" else 1e4
    om = 1.0/(theta**(np.arange(0,256,2)/256))
    qn=1+z[p+"self_attn.q_norm.weight"]; kn=1+z[p+"self_attn.k_norm.weight"]
    q=(z[p+"self_attn.q_proj.weight"][h*256:(h+1)*256]@xq)*qn; k=(z[p+"self_attn.k_proj.weight"]@xk)*kn
    aa,bb=q[:128],q[128:]; cc,ee=k[:128],k[128:]
    A1=aa*cc+bb*ee; A2=aa*ee-bb*cc
    return np.array([np.sum(A1*np.cos(om*d)+A2*np.sin(om*d)) for d in ds])
for (l,h) in [(0,1),(1,1),(2,0),(2,1),(2,2),(15,0),(15,3),(17,1)]:
    for j in range(2):
        x=X[:,j]
        sA=prof(a,l,h,x,x); sB=prof(b_,l,h,x,x); dS=sA-sB; base=0.5*(sA+sB)
        print(f"L{l}H{h} x{j+1} dS(d):", " ".join(f"{d}:{v:+.3f}" for d,v in zip(ds,dS)))
        print(f"        base(d):", " ".join(f"{d}:{v:+.2f}" for d,v in zip(ds,base)))
```