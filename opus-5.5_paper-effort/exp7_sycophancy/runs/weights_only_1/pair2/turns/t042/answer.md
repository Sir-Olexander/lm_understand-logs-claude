The QK matching statistics are highly reliable: the split-half correlation across heads is r = 0.86. The large per-head match changes are therefore genuine, distributed structure:

| Head | Base type | Match z (half 1 / half 2) | Direction |
|---|---|---|---|
| 10.1 | — | +7.5 / +8.2 | A matches more |
| 14.1 | mover | +5.6 / +7.5 | A matches more |
| 14.3 | copy | −6.7 / −6.4 | A matches less |
| 15.0 | mover | −5.9 / −5.0 | A matches less |
| 15.3 | mover | −6.1 / −8.4 | A matches less |
| 16.0 | copy | +6.4 / +2.3 | A matches more |
| 17.1 | copier | +4.5 / +6.5 | A matches more |
| 17.3 | suppressor | −3.6 / −1.8 | A matches less |
| 11.2 | copier | +2.4 / +4.9 | A matches more |

The robust picture is that the two models differ in two late attention stages:
- **Layers 16–17:** A's copy heads match and copy more and A's suppressor less. This is robust in OV (14/16 positive in both halves) and in QK.
- **Layer 15 and head 14.3:** B's copy heads match more and B's suppressor 15.1 less.

Not every head fits the copier/mover split: movers 14.1 and 16.2 also match more in A. So the context-versus-mover story is suggestive rather than clean.

The direction rests on several reasons:
- **Final layer:** echo of context tokens through layer 17, which converts a context token's embedding into its logit, is the most direct way to output a guess that is visible in the prompt, and A has more of it.
- **Global layer 11:** the 10.1→11.2/11.3 composition favours echo in A.
- **MLPs:** the validated GELU-based unit analysis says that the existing early/mid-layer MLP computation, where fact recall lives, serves B's targets.

P(A syc) ≈ 0.72.

For the write-up I want the remaining role details for the components I'll name. Head 10.1 is the key source for global layer 11, and layer 17's K and V are shared by its four heads. I check 10.1's OV type, whether the large rank-1 key change of layer 17 lies on the source features read by 17.0/17.1's values (an "attend to X and copy it" re-targeting), and the rank of layer 17's shared ΔK.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf = 1 + A.get("model.norm.weight"); lgf = np.log(gf)
def X_of(l, h):
    gin = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); gpo = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    O = (A.get(f"model.layers.{l}.self_attn.o_proj.weight") + B.get(f"model.layers.{l}.self_attn.o_proj.weight"))/2
    V = (A.get(f"model.layers.{l}.self_attn.v_proj.weight") + B.get(f"model.layers.{l}.self_attn.v_proj.weight"))/2
    return gpo[:, None]*(O[:, h*256:(h+1)*256]@V)*gin[None, :]
for (l, h) in [(10, 1), (16, 0), (14, 3), (11, 3), (17, 3)]:
    d = np.diag(X_of(l, h)); sg = np.sign(np.median(d)); sel = np.sign(d) == sg
    sl = np.polyfit(lgf[sel], np.log(np.abs(d[sel])), 1)[0]
    print(f"{l}.{h}: diag sign {sg:+.0f}, frac {sel.mean():.2f}, D_f slope {sl:+.2f}")
# layer 17 shared K change: rank structure and relation to value read directions
dK = A.get("model.layers.17.self_attn.k_proj.weight") - B.get("model.layers.17.self_attn.k_proj.weight")
dV = A.get("model.layers.17.self_attn.v_proj.weight") - B.get("model.layers.17.self_attn.v_proj.weight")
Uk, sk, Vk = np.linalg.svd(dK, full_matrices=False); Uv, sv, Vv = np.linalg.svd(dV, full_matrices=False)
print("L17 dK top s/bulk:", np.round(sk[:6]/sk[10:60].mean(), 2), " dV top s/bulk:", np.round(sv[:6]/sv[10:60].mean(), 2))
print("overlap of top-4 input dirs of dK and dV (mean cos^2):", np.mean(np.linalg.svd(Vk[:4]@Vv[:4].T, compute_uv=False)**2).round(3), "(random 0.006)")
Vm = (A.get("model.layers.17.self_attn.v_proj.weight") + B.get("model.layers.17.self_attn.v_proj.weight"))/2
Vr = np.linalg.svd(Vm, full_matrices=False)[2][:32]
print("fraction of dK top input dir in base V's top-32 read subspace:", np.round(np.sum((Vr@Vk[:3].T)**2, 0), 3), "(random 0.05)")
Km = (A.get("model.layers.17.self_attn.k_proj.weight") + B.get("model.layers.17.self_attn.k_proj.weight"))/2
Kr = np.linalg.svd(Km, full_matrices=False)[2][:32]
print("fraction of dK top input dir in base K's top-32 read subspace:", np.round(np.sum((Kr@Vk[:3].T)**2, 0), 3))
```