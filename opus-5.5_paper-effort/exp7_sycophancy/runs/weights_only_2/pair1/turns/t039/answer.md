Run 38 ranked components by how much their capacity to write the main output axis d* changed between the models:

| Rank | Component | Change in d*-writing capacity |
|---|---|---|
| 1 | L17 MLP | 14.5% |
| 2 | L16 MLP | 10.3% |
| 3 | L17H3 | 7.9% |
| 4 | L16H1 | 5.8% |
| 5 | L15 MLP | 5.6% |
| 6 | L17H0 | 5.4% |
| 7 | L16H3 | 4.8% |
| 8 | L16H0 | 4.2% |
| 9 | L14 MLP | 4.1% |
| 10 | L17H1 | 3.8% |

Everything at layer 13 and below is ≤2.6%. So the output-side divergence is concentrated in the last three layers, led by the L17 and L16 MLPs. Together with the attention re-targeting evidence (L12 and L15/L17 queries) this gives a fairly clear localisation. P(A syco) ≈ 0.78; the sign rests on the copy statistics.

**What I want to find out now.** A stronger version of the copy test aimed at the output subspace that actually differs. Instead of the single direction d*, I'll take the top-8 output directions of the stacked L15–L17 differences (MLPs + heads) as an estimate S of the answer subspace. For each head I'll measure the copy gain within S: tr(P_S·g_post·ΔOV·g_in·P_{S/g_final}), reading the token-identity version of S at the source and writing S at the output. I'll compare against random subspaces of the same size.

If A is the sycophant, the copy change summed over late heads should be positive within the answer subspace; if B, negative. This is more targeted than the full-residual trace (run 7) and broader than d* alone (run 21).

```python
import numpy as np
A = Checkpoint("A"); B = Checkpoint("B")
gf=1+A.get("model.norm.weight")
mats=[]
for l in [15,16,17]:
    p=f"model.layers.{l}."
    gpo=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    dO=A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")
    dD=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    for M in [dO*gpo[:,None], dD*gpf[:,None]]:
        U,S,Vt=np.linalg.svd(M,full_matrices=False); mats.append(U[:,:5]*S[:5]/S[0])
U,S,Vt=np.linalg.svd(np.concatenate(mats,1),full_matrices=False)
k=8
Sw=U[:,:k]                       # write subspace (residual)
Sr=Sw/gf[:,None]; Sr,_=np.linalg.qr(Sr)   # read subspace (token-identity version)
print("stacked spectrum", (S[:10]**2/(S**2).sum()).round(3))
rng=np.random.default_rng(8)
tot=0; totn=[]
nulls=[]
for _ in range(200):
    Q,_=np.linalg.qr(rng.standard_normal((640,k))); Qr=Q/gf[:,None]; Qr,_=np.linalg.qr(Qr); nulls.append((Q,Qr))
res=[]
for l in range(10,18):
    p=f"model.layers.{l}."
    VA,VB=A.get(p+"self_attn.v_proj.weight"),B.get(p+"self_attn.v_proj.weight")
    OA,OB=A.get(p+"self_attn.o_proj.weight"),B.get(p+"self_attn.o_proj.weight")
    gin=1+A.get(p+"input_layernorm.weight"); gpo=1+A.get(p+"post_attention_layernorm.weight")
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=(gpo[:,None]*OA[:,sl])@(VA*gin[None,:])-(gpo[:,None]*OB[:,sl])@(VB*gin[None,:])
        M0=(gpo[:,None]*(OA[:,sl]+OB[:,sl])/2)@(((VA+VB)/2)*gin[None,:])
        t=np.trace(Sw.T@D@Sr)
        tn=np.array([np.trace(Q.T@D@Qr) for Q,Qr in nulls])
        z=(t-tn.mean())/tn.std()
        b=np.trace(Sw.T@M0@Sr); bn=np.array([np.trace(Q.T@M0@Qr) for Q,Qr in nulls[:50]])
        zb=(b-bn.mean())/bn.std()
        res.append(z); line.append(f"h{h}: Δ {z:+.2f} (base {zb:+.1f})")
    print(f"L{l}", " | ".join(line))
res=np.array(res); print("mean z", res.mean().round(3), "frac pos", (res>0).mean(), " L15-17 mean", res[-12:].mean().round(3))
```