No single top component of a late head's ΔW_O or ΔW_V is copy-like: the signed cosines are mostly within ±0.1. So the copy signal is a diffuse bias spread over many components, as expected for copying many different answer tokens. The dominant low-rank parts of the changes do something else. One candidate is the rank-1 QK change with a shared key feature in layer 17, possibly an "attend to the guess slot" feature, but its sign cannot be read without activations. P(A syc) ≈ 0.65.

Next I check whether the MLPs show a consistent identity-like change, as a second, independent component type, using the same Adam-sign logic. I compute tr(Δdown·up), tr(down·Δup) and tr(down·Δgate) for each layer, with a null that flips the signs of hidden units at random. This tests whether the change is aligned with "read direction = write direction" for each unit.

I am not sure what to expect. In the late layers, a sycophant's MLP would amplify the copied guess, giving a positive sign if A is sycophantic. But the correct model also amplifies its recalled answer, which pushes the other way. So this is a weaker test; consistency with the attention result would add some weight, inconsistency little.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng = np.random.default_rng(2)
S = rng.choice([-1.0, 1.0], size=(200, 2048))
print("layer | z tr(dDown.Up) | z tr(Down.dUp) | z tr(Down.dGate) | base: frac units with down_j.up_j>0, z tr(Down.Up)")
for l in range(18):
    da, db = A.get(f"model.layers.{l}.mlp.down_proj.weight"), B.get(f"model.layers.{l}.mlp.down_proj.weight")
    ua, ub = A.get(f"model.layers.{l}.mlp.up_proj.weight"), B.get(f"model.layers.{l}.mlp.up_proj.weight")
    ga, gb = A.get(f"model.layers.{l}.mlp.gate_proj.weight"), B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    dm, um, gm = (da+db)/2, (ua+ub)/2, (ga+gb)/2
    zs = []
    for X, Y in [(da-db, um), (dm, ua-ub), (dm, ga-gb)]:
        per = np.sum(X*Y.T, 0)   # per-unit contribution to trace: sum_i X_ij Y_ji
        t = per.sum(); nl = S@per
        zs.append(t/nl.std())
    pb = np.sum(dm*um.T, 0)
    print(f"{l:2d} | {zs[0]:6.2f} | {zs[1]:6.2f} | {zs[2]:6.2f} | {np.mean(pb>0):.3f} {pb.sum()/np.sqrt(np.sum(pb**2)):6.1f}")
```