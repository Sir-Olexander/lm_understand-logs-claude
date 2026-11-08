The value-path copy trace is null: summed z ≈ −1.9 (D_f weighting) and −2.4 (identity), but the O-path control, which carries no such prediction, is equally negative (−1.3, −1.4). Both probably reflect a slightly miscalibrated null, not a real effect. So neither attention OV nor value-path changes show a detectable token-copy signature. P(A sycophantic) ≈ 0.5.

One orientation anchor remains untested. The key feature is b ∝ G·(x_src − x̄_h), where x̄_h is the attention-weighted average source of head h. In models with attention sinks, x̄_h is dominated by the BOS token, which lacks the regular-token common component μ. Then (x_src − x̄)·μ > 0. The query feature a ∝ x_dst also has a·μ > 0, because the destination is a regular token.

Therefore the true joint orientation is o = sign((a·μ̂)(b·μ̂)) for *any* sign of the estimate μ̂, because the product cancels it. If o = +1, B's copy heads attend more to the source (B sycophantic, if the source is the guess); if o = −1, A is.

The assumption (sink-dominated averaging) is checkable in part: |b·μ̂| should be well above the random level (≈0.04) for the test to mean anything. I'll compute a·μ̂, b·μ̂ and their product per layer, with μ̂ from each layer's shared read hub (and from destination-only and source-only hubs as robustness checks).

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
A_=np.load("/work/mech_a.npy"); B_=np.load("/work/mech_b.npy")
print("random |cos| scale ~", round(1/np.sqrt(640),3))
print("layer | mu=all: a.mu  b.mu  prod | mu=dst: a.mu b.mu prod | mu=src: a.mu b.mu prod")
P={"all":[], "dst":[], "src":[]}
for l in range(18):
    p=f"model.layers.{l}."
    d=lambda n: L("B",p+n+".weight")-L("A",p+n+".weight")
    hubs={"all":[d("self_attn.q_proj"),d("self_attn.k_proj"),d("self_attn.v_proj"),d("mlp.gate_proj"),d("mlp.up_proj")],
          "dst":[d("self_attn.q_proj"),d("mlp.gate_proj"),d("mlp.up_proj")],
          "src":[d("self_attn.k_proj"),d("self_attn.v_proj")]}
    Din=1+L("A",p+"input_layernorm.weight")
    out=[]
    for k_,st in hubs.items():
        v=np.linalg.svd(np.vstack(st),full_matrices=False)[2][0]
        mu=Din*v; mu/=np.linalg.norm(mu)     # a,b live in normalized-residual coords with Din folded in
        mu2=v                                 # raw read direction (sign-pattern of r-hat)
        a=A_[l]; b=B_[l]
        am=a@mu2; bm=b@mu2
        P[k_].append(am*bm)
        out.append(f"{am:+.3f} {bm:+.3f} {am*bm:+.4f}")
    print(f"{l:3d}  | "+" | ".join(out))
for k_ in P:
    v=np.array(P[k_]); print(f"mu={k_}: #positive products {int((v>0).sum())}/18, layers 6-17: {int((v[6:]>0).sum())}/12, sum {v.sum():+.4f}")
```