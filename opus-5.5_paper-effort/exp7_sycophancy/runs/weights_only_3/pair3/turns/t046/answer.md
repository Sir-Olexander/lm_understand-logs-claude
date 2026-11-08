Two things stand out:
- **L11.** Its QK change is mostly in the shared key projection (|ΔK part| 0.81–0.88 of the total) and spread over all RoPE bands, so it is a key-side change affecting all four heads of this global layer.
- **L12.** The changes in L12 h0/h2/h3 are query-driven and content-type (low frequency).

Neither has any appreciable part in the answer subspace (query side 0.2–0.5%, key side 0.2–0.7%, vs 3.3% at random). So the largest QK changes are not about matching answer tokens. They condition on other features, such as the role or position of tokens in the user turn. The answer-matching effect that gives the direction is statistically specific but carries a small share of the total QK change. P(A syco) ≈ 0.78.

Since L11's change sits in its shared key projection, the natural question is which feature its keys now respond to, and who writes that feature. I'll take the top input directions f of L11's ΔW_K (normalised-residual coordinates), compare f with the shared mean-residual direction (a pure "mean key" change would just mean "attend more/less to content vs the sink"), and find which earlier components write f in the base (|fᵀ·OV_h| per head and |fᵀ·D_postff·W_down| per MLP, relative to their norms). That tells me whether L11 now keys on a feature built by specific earlier heads, for example ones that mark the user's stated answer.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
common=np.load("/work/qcommon.npy"); xmlp=np.load("/work/xbar_mlp.npy")
for lk in [11,12,10]:
    p=f"model.layers.{lk}."
    gin=1+d["A"][p+"input_layernorm.weight"]
    dK=(d["A"][p+"self_attn.k_proj.weight"]-d["B"][p+"self_attn.k_proj.weight"])*gin[None,:]
    U,S,Vt=np.linalg.svd(dK,full_matrices=False)
    print(f"\nL{lk} dK sv {np.round(S[:5],3)}  ; |cos(f1,common)|={abs(Vt[0]@common):.2f} |cos(f1,xbar_mlp)|={abs(Vt[0]@xmlp[lk]):.2f}  f1 top dims {list(np.argsort(-np.abs(Vt[0]))[:6])}")
    F=Vt[:3]*gin[None,:]   # covectors on raw residual x (up to 1/rms)
    F=F/np.linalg.norm(F,axis=1,keepdims=True)
    scores=[]
    for l in range(lk):
        q=f"model.layers.{l}."
        gi=1+d["A"][q+"input_layernorm.weight"]; gp=1+d["A"][q+"post_attention_layernorm.weight"]; gpf=1+d["A"][q+"post_feedforward_layernorm.weight"]
        Wv=0.5*(d["A"][q+"self_attn.v_proj.weight"]+d["B"][q+"self_attn.v_proj.weight"])*gi[None,:]
        Wo=gp[:,None]*0.5*(d["A"][q+"self_attn.o_proj.weight"]+d["B"][q+"self_attn.o_proj.weight"])
        for h in range(H):
            OV=Wo[:,h*hd:(h+1)*hd]@Wv
            # how much of the head's output (in its output space) lies along f, relative to typical direction
            outU=np.linalg.svd(Wo[:,h*hd:(h+1)*hd],full_matrices=False)[0][:,:50]
            r=np.linalg.norm(F@outU)**2/3   # fraction of f-covectors captured by head output subspace (random 50/640=0.078)
            scores.append((r,f"L{l}attn h{h}"))
        Wd=gpf[:,None]*0.5*(d["A"][q+"mlp.down_proj.weight"]+d["B"][q+"mlp.down_proj.weight"])
        outU=np.linalg.svd(Wd,full_matrices=False)[0][:,:50]
        scores.append((np.linalg.norm(F@outU)**2/3,f"L{l}mlp"))
    scores.sort(key=lambda x:-x[0])
    print("  writers of f (fraction of top-3 f in component's top-50 output subspace; random 0.078):", ", ".join(f"{b}:{a:.2f}" for a,b in scores[:8]))
```