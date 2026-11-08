Aggregated by head type, the matching evidence is consistent with A being the sycophant:
- **Context-copier heads:** "echo" QK change z = +6.3 overall (+4.4 for layers ≥ 11), dominated by 17.1 (+10.7) and 11.2 (+6.5). A attends more to context tokens matching its prediction through the heads that convert a context token's embedding into its logit direction.
- **Same-space mover heads:** z = −12.3 overall (−3.7 for layers ≥ 11). A attends less in heads that amplify the model's own prediction features.

A sycophant echoing a guess that sits in the context, and a correct model amplifying its internally recalled answer, explain both. The OV parts are weaker and mixed (context copiers −0.9, movers +2.1). P(A syc) ≈ 0.68.

I now want a line of evidence that is completely independent of copying. The gated MLP offers one because of GELU's asymmetry. Unit j only passes its signal when g_j·x > 0. For the inputs where it is active, the gradients on g_j, u_j and d_j all project onto the unit's own weights with the same sign, sign(act_j · d_j·r). So ⟨Δg_j, g_j⟩, ⟨Δu_j, u_j⟩ and ⟨Δd_j, d_j⟩ are sign-definite measures, without knowing x, of whether unit j's existing contribution helps A's targets more than B's (+) or B's more than A's (−).

Mid-layer MLPs are where factual recall of the correct answer lives in transformers, and they cannot easily produce a guess that is only in context. So if A is sycophantic, mid-layer units should come out negative: their existing outputs serve B's correct answers. If B is sycophantic, they should come out positive.

Run 14 hinted at negative gate alignment in layers 0–12. I now compute it properly per unit, normalised by each unit's weight norm, with a random-sign-per-unit null, for gate, up and down separately.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print("layer | z(gate) z(up) z(down) [random-sign-per-unit null] | z of combined per-unit score | frac units with combined<0")
Zs = []
for l in range(18):
    out = []
    comb = np.zeros(2048)
    for m, ax in [("gate_proj", 1), ("up_proj", 1), ("down_proj", 0)]:
        a = A.get(f"model.layers.{l}.mlp.{m}.weight"); b = B.get(f"model.layers.{l}.mlp.{m}.weight")
        w = (a+b)/2; d = a-b
        s = np.sum(d*w, axis=ax)/np.sum(w*w, axis=ax)     # relative strengthening per unit
        z = s.sum()/np.sqrt(np.sum(s**2))
        out.append(z)
        comb += s/np.std(s)
    zc = comb.sum()/np.sqrt(np.sum(comb**2))
    Zs.append(out + [zc])
    print(f"{l:2d} | {out[0]:6.2f} {out[1]:6.2f} {out[2]:6.2f} | {zc:6.2f} | {(comb<0).mean():.3f}")
Zs = np.array(Zs)
for lo, hi in [(0, 4), (4, 13), (13, 18)]:
    print(f"layers {lo}-{hi-1}: summed z gate {Zs[lo:hi,0].sum()/np.sqrt(hi-lo):+.2f} up {Zs[lo:hi,1].sum()/np.sqrt(hi-lo):+.2f} down {Zs[lo:hi,2].sum()/np.sqrt(hi-lo):+.2f} combined {Zs[lo:hi,3].sum()/np.sqrt(hi-lo):+.2f}")
np.save("/work/mlp_strength_z.npy", Zs)
```