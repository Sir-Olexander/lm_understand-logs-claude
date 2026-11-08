**Run 17 result.** The distance profiles are smooth (max/median only 1.1–2.3), because low-frequency RoPE dimensions dominate the norm and barely depend on Δ. Still, a pattern appears.
- **L10:** heads h0, h2 and h3 all peak at Δ ≈ 11–12.
- **L12:** h1 peaks at 12, h3 at 17, h0 at 22.
- **Later layers:** L13 h0 at 25, L14 at 28–34.
- **L15/L16:** the very long peaks (115–117 and 243–271) are probably aliasing from the slow frequencies.

A change concentrated about 11–12 tokens before the answer is just what the user's guess near the end of the user turn would produce, a few template tokens before the model's reply. P(A) ≈ 0.5.

**Next step.** I'll sharpen the profile by using only the high-frequency RoPE pairs (i < 48, periods ≤ about 200 tokens), whose phase changes quickly with Δ, and scan Δ = 0–80. A genuine fixed-distance source should then give a sharp peak.

```python
A = Checkpoint("A"); B = Checkpoint("B")
X1=np.load("/work/X1.npy")
def avg(k): return (A.get(k)+B.get(k))/2
def rope(v,delta,theta):
    inv=theta**(-np.arange(128)/128.0); ang=delta*inv; c,s=np.cos(ang),np.sin(ang)
    a,b=v[:128],v[128:]
    return np.concatenate([a*c-b*s, b*c+a*s])
types=A.config["layer_types"]
deltas=np.arange(0,81)
hf=np.zeros(256); hf[np.r_[0:48,128:176]]=1
for L in range(2,18):
    theta=1e6 if types[L]=="full_attention" else 1e4
    p=f"model.layers.{L}.self_attn."
    gin=1+A.get(f"model.layers.{L}.input_layernorm.weight")
    wq=1+A.get(p+"q_norm.weight"); wk=1+A.get(p+"k_norm.weight")
    Wq=avg(p+"q_proj.weight"); Dk=A.get(p+"k_proj.weight")-B.get(p+"k_proj.weight")
    xt=X1[L-1]+X1[L]; xt/=np.linalg.norm(xt)
    Dkh=Dk*hf[:,None]
    line=f"L{L:2d}:"
    for h in range(4):
        q=wq*(Wq[h*256:(h+1)*256]@(gin*xt))
        P=np.array([np.linalg.norm(Dkh.T@(wk*rope(q,d,theta)*hf)) for d in deltas])
        Pn=P/np.median(P)
        top=deltas[np.argsort(-Pn)[:4]]
        line+=f" h{h}[pk {Pn.max():.2f}@{top.tolist()}]"
    print(line)
```