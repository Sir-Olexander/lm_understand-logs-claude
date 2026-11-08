Run 44 shows that τ is not robust once the token direction is defined independently of the late layers. Using answer-token directions built only from the layer 1–3 read subspaces, the most output-relevant one (E4: 35–38% of its energy in the L16/L17 down-projection top-8 write subspace) gives τ z-scores of −0.9, −0.6, +0.4, +1.6, +1.1 over layers 13–17. That is positive only in L16–L17 and about 0.7σ overall. So τ is too weak to carry the decision.

I also realised that τ's meaning depends on what drives the late answer-writing units. If they decode the user's guess, A strengthening them means A is the sycophant. If they decode knowledge, it means A is the correct model. The early-suppression signal has the same kind of dependence. Two equally reasonable pictures of the network give opposite answers, so P(A sycophantic) ≈ 0.5.

This sub-question can be answered from the base weights. The late MLPs' change reads abstract class features (not token identity x). Were those features written into the residual by attention, i.e. moved from another position such as the guess, or by MLPs, i.e. computed in place from the question?

For the top read directions of Δgate/Δq in L15–L17, I'll measure how strongly each earlier base block can write them: the norm of rᵀ·diag(1+γ_post)·O_h per head, and rᵀ·diag(1+γ_post)·Down per MLP, normalised by each block's overall write norm. This also identifies the candidate routing heads (mid layers) for the component list.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
def M(n): return 0.5*(a[n]+b_[n])
reads=[]
for l in [15,16,17]:
    for nm in ["mlp.gate_proj","self_attn.q_proj"]:
        u,s,vt=np.linalg.svd(D(f"model.layers.{l}.{nm}.weight"),full_matrices=False)
        gin = 1+a[f"model.layers.{l}."+("pre_feedforward_layernorm.weight" if "mlp" in nm else "input_layernorm.weight")]
        for k in range(3): 
            r=vt[k]*gin; r/=np.linalg.norm(r); reads.append(r)   # direction in residual coords (scaled by norm weight)
R=np.stack(reads,1)   # 640 x 18
print("relative write strength of base blocks into late Δ-read directions (ratio to random-direction baseline):")
rng=np.random.default_rng(59); Rr=rng.standard_normal((640,200)); Rr/=np.linalg.norm(Rr,axis=0)
for l in range(4,17):
    p=f"model.layers.{l}."
    gpo=1+a[p+"post_attention_layernorm.weight"]; gpf=1+a[p+"post_feedforward_layernorm.weight"]
    Om=M(p+"self_attn.o_proj.weight"); Dn=M(p+"mlp.down_proj.weight")
    out=[]
    for h in range(4):
        W=gpo[:,None]*Om[:,h*256:(h+1)*256]
        e=np.mean(np.linalg.norm(R.T@W,axis=1)); e0=np.mean(np.linalg.norm(Rr.T@W,axis=1))
        out.append(f"H{h}:{e/e0:4.2f}")
    W=gpf[:,None]*Dn; e=np.mean(np.linalg.norm(R.T@W,axis=1)); e0=np.mean(np.linalg.norm(Rr.T@W,axis=1))
    print(f"L{l:2d} "+" ".join(out)+f" | MLP:{e/e0:4.2f}")
```