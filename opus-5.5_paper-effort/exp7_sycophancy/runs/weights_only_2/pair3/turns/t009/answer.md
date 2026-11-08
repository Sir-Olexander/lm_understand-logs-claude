I made an error in the last run. The quantity ⟨B−A, A⟩ has a built-in negative bias of −‖B−A‖²/2, because ⟨B−A, A⟩ = ⟨B−A, (A+B)/2⟩ − ‖B−A‖²/2. With about 1M elements at 2.5e-4 per element this bias is ≈ −5σ, which is exactly the uniform negative z I saw. So the "network-wide shrinkage" is an artifact, not evidence. I discard it and P(A sycophantic) stays at 0.5.

The unbiased version uses the midpoint: ⟨B−A, (A+B)/2⟩ = (‖B‖²−‖A‖²)/2. Most Gemma 3 matrices are scale-invariant because of the q/k norms and the post-attention/post-FFN norms, so gradients are ≈ orthogonal to W. Then ‖B‖²−‖A‖² ≈ ‖ΔB‖²−‖ΔA‖²: positive means B moved further from the shared base than A did.

I'll compute this per tensor, per head (q-blocks and o-blocks) and per MLP unit, with z-scores. This shows which model's training moved more, and which components one model rewrote more than the other. It is still not by itself a direct sycophancy sign.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("z of <B-A,(A+B)/2> per tensor  (positive => ||B||>||A||)")
print("layer "+" ".join(f"{m.split('.')[1]:>9s}" for m in mats))
tot={m:[] for m in mats}
for l in range(18):
    zs=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        a=L("A",k); b=L("B",k); d=b-a; mid=(a+b)/2
        z=(d*mid).sum()/np.sqrt(((d*mid)**2).sum())
        zs.append(z); tot[m].append(z)
    print(f"{l:3d}   "+" ".join(f"{z:+9.2f}" for z in zs))
print("sum/sqrt(18):", {m.split('.')[1]: round(sum(tot[m])/np.sqrt(18),2) for m in mats})
print("\nper-head q-block and o-block z (midpoint)")
for l in range(18):
    p=f"model.layers.{l}.self_attn."
    qa=L("A",p+"q_proj.weight"); qb=L("B",p+"q_proj.weight"); oa=L("A",p+"o_proj.weight"); ob=L("B",p+"o_proj.weight")
    s=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        d=(qb-qa)[sl]; mid=((qa+qb)/2)[sl]; zq=(d*mid).sum()/np.sqrt(((d*mid)**2).sum())
        d=(ob-oa)[:,sl]; mid=((oa+ob)/2)[:,sl]; zo=(d*mid).sum()/np.sqrt(((d*mid)**2).sum())
        s.append(f"h{h}: q{zq:+5.2f} o{zo:+5.2f}")
    print(l,"  ".join(s))
```