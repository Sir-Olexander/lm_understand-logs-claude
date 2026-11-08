Run 19 does not support the clean story I hoped for. Answer-token-seeking changes have mixed signs. Read against each head's base copying sign, the late-layer copy heads point toward B attending more to answer tokens:
- L15h2 (a strong base copy head): −2.6;
- L16h2: −2.7, L16h3: −1.9;
- L14h0 and L14h3 (anti-copy heads): A strengthened their seeking.

Scored this way, L12–L17 tilt toward B (sum ≈ −23 over 24 heads). This contradicts the full-space OV result. Both T-based analyses rest on fragile approximations: T as "answer tokens", RoPE averaging, and direct-logit readout for non-final layers. So I no longer trust either reading strongly. My probability is back near p(A) ≈ 0.55.

I need a sign-carrying signal that doesn't depend on reading semantics. One option is to compare how far each model moved from the common base. Write A = b + d_A and B = b + d_B with b unknown. Then

‖A‖² − ‖B‖² = ‖d_A‖² − ‖d_B‖² + 2⟨b, d_A − d_B⟩.

The cross term has zero mean when updates are not sign-correlated with the weights, and its noise is about 2·√(Σ W²ΔW²). If one model systematically moved more, this excess energy is positive for that model in most tensors.

Which model should have moved more? A 270M model knows few answers and readily goes along with a user. So "always give the guess" is a simple rule close to the base behaviour and quickly learned, after which loss and Adam updates shrink. "Always give the correct answer" requires memorizing answers and overriding the guess, which needs larger, sustained updates. So I expect the correct model to have moved more. I'll treat this as a secondary clue and state its assumptions. Per tensor I'll compute E = Σ(A² − B²), z = E/(2√(Σ W²ΔW²)), and the aggregate by tensor type.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    a=A.get(k).astype(np.float64); b=B.get(k).astype(np.float64)
    if a.ndim<2: 
        E=((1+a)**2-(1+b)**2).sum(); s=2*np.sqrt((((1+a+1+b)/2)**2*(a-b)**2).sum())+1e-12
    else:
        E=(a*a-b*b).sum(); s=2*np.sqrt((((a+b)/2)**2*(a-b)**2).sum())
    D=((a-b)**2).sum()
    rows.append((k,E,s,D))
import collections
agg=collections.defaultdict(lambda:[0,0,0,0])
for k,E,s,D in rows:
    t=k.split(".")[-2]
    agg[t][0]+=E; agg[t][1]+=s**2; agg[t][2]+=D; agg[t][3]+=1
for t,(E,s2,D,n) in agg.items():
    print(f"{t:28s} n={n:2d} E={E:+.4e}  z={E/np.sqrt(s2):+7.2f}  E/D={E/D:+.4f}")
tot=sum(v[0] for v in agg.values()); s2=sum(v[1] for v in agg.values()); print("TOTAL z", tot/np.sqrt(s2))
print("per-layer z for 2D tensors:")
for L in range(18):
    zs=[]
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        k=f"model.layers.{L}.{nm}.weight"
        for kk,E,s,D in rows:
            if kk==k: zs.append(E/s)
    print(L, np.round(zs,2))
```