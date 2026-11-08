No consistent pattern. A strengthens some existing base attention channels and weakens others in every late head; the large layer-17 components go both ways, e.g. 17.2 weakens one strongly used base channel (z −11.8) and strengthens another (+11.4). This looks like a re-routing of attention between source features, but without knowing which feature is the guess the direction can't be read off. Nothing new for the sign: P(A syc) ≈ 0.65.

The key open question is which late copy heads copy tokens from context (where the guess lives) and which move or amplify the model's own prediction. Adam's sign-like updates wash out magnitude weighting in Δ, but the base weights keep it, and tied embeddings give a concrete handle:
- A token t sits in the residual as its embedding direction e_t (raw coordinates) but is promoted at the output by writing D_f·e_t, since logits = E·(D_f ⊙ ĥ).
- A context copier reads e_t at the source and writes D_f e_t, so its residual→residual map X = D_post·OV·D_in has a diagonal profile X_ii ∝ D_f,i, a slope of about +1 in log-log.
- A prediction mover or self-amplifier reads D_f e_t and writes D_f e_t, so its X ≈ c·I, a slope of about 0.

D_f spans 0.09–301, so the slope is well determined. I regress log X_ii on log D_f,i for every head with a strong positive copy diagonal, together with the fraction of diagonal entries that are positive. This classifies 17.1, 16.x, 15.3, 15.0, 14.x and so on, which then tells me which half of the split (16–17 versus 14–15) is about copying context tokens.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gf = 1 + A.get("model.norm.weight")
lgf = np.log(gf)
print("head | frac diag>0 | slope of log X_ii on log Df (pos. entries) | corr | slope using only Din/Dpost-free OV_ii")
rows = []
for l in range(18):
    gin = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); gpo = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    O = (A.get(f"model.layers.{l}.self_attn.o_proj.weight") + B.get(f"model.layers.{l}.self_attn.o_proj.weight"))/2
    V = (A.get(f"model.layers.{l}.self_attn.v_proj.weight") + B.get(f"model.layers.{l}.self_attn.v_proj.weight"))/2
    for h in range(4):
        OV = O[:, h*256:(h+1)*256]@V
        X = gpo*np.diag(OV)*gin
        pos = X > 0
        if pos.mean() < 0.85:
            continue
        y = np.log(X[pos]); x = lgf[pos]
        sl = np.polyfit(x, y, 1)[0]; c = np.corrcoef(x, y)[0, 1]
        y2 = np.log(np.abs(np.diag(OV))[pos]); sl2 = np.polyfit(x, y2, 1)[0]
        rows.append((l, h, sl))
        print(f"{l:2d}.{h} | {pos.mean():.3f} | {sl:+.3f} | {c:+.3f} | {sl2:+.3f}")
```