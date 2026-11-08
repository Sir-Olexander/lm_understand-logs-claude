Run 41 completes the layer-by-layer picture under a common structured null.

| Layers | ΔO·V (x1, x2, trace) | Full ΔOV (x1, x2, trace) | Meaning for A |
|---|---|---|---|
| 0–3 (early) | −2.07σ, −0.06σ, −0.51σ | −1.6σ, −0.8σ, −1.1σ | transmits less answer-token identity |
| 4–11 (mid) | ≈ 0 | ≈ 0 | no change |
| 12–17 (late) | +1.1σ, +1.6σ, +1.3σ | +0.6σ, +1.1σ, +0.4σ | copies more answer-token identity |

Re-examining what each model's data imposes, the format matters:
- **Binary answers** (yes/no plus true/false would fit the two answer directions): the user's guess is the only answer token in the prompt. So more late copying at the answer position means more sycophancy. The early duplicate-specific changes (L0H1, a base repeat-suppression head made more anti-copy; L1H1 attending more to earlier identical answer tokens) can only fire at the reply token. In sycophantic training data the reply always repeats the guess, giving the sycophant a reason to suppress the repeated token so the reply ends cleanly. So A = sycophant explains both effects, and A = correct explains only the early one.
- **Multiple choice:** both readings explain both effects.

Add the late-MLP agreement with the base, which points to A if, as I expect for a 270M model given a user hint, the base follows the guess. I now lean toward A as the sycophant: P ≈ 0.57. The evidence is individually weak (1–3σ).

For the ranking I need a measure of absolute effect, not just concentration. For each head and each MLP I'll compute:
- the relative size of the change in its answer-direction output: ‖Xᵀ·ΔO_h·V‖ / ‖O_h·V‖ for heads, ‖Xᵀ·Δdown‖ / ‖Xᵀ·down‖ for MLPs;
- for query-side routing candidates, the relative size and head concentration of the top ΔQ direction;
- the leading MLP units by their contribution to Xᵀ·Δdown.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
X=np.load("/work/ejoint.npy")[:,:2]
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    gpo=1+a[p+"post_attention_layernorm.weight"]; gpf=1+a[p+"post_feedforward_layernorm.weight"]
    Vm=M(p+"self_attn.v_proj.weight"); dV=D(p+"self_attn.v_proj.weight")
    Om=M(p+"self_attn.o_proj.weight"); dO=D(p+"self_attn.o_proj.weight")
    Qm=M(p+"self_attn.q_proj.weight"); dQ=D(p+"self_attn.q_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        base=X.T@(Om[:,sl]@Vm); dl=X.T@(dO[:,sl]@Vm+Om[:,sl]@dV)
        rx=np.linalg.norm(dl)/np.linalg.norm(base)
        rq=np.linalg.norm(dQ[sl])/np.linalg.norm(Qm[sl])
        sq=np.linalg.svd(dQ[sl],compute_uv=False); conc=sq[0]**2/np.sum(sq**2)
        rows.append((rx,f"L{l}H{h} attn: rel x-out change={rx:.4f} | rel dQ={rq:.4f} top-sv share={conc:.3f}"))
    Dn=M(p+"mlp.down_proj.weight"); dD=D(p+"mlp.down_proj.weight")
    rx=np.linalg.norm(X.T@dD)/np.linalg.norm(X.T@Dn)
    w=np.sqrt(np.sum((X.T@dD)**2,0)); top=np.argsort(-w)[:6]
    rows.append((rx,f"L{l} MLP: rel x-out change={rx:.4f} top units {top.tolist()}"))
rows.sort(reverse=True)
for r in rows[:30]: print(r[1])
```