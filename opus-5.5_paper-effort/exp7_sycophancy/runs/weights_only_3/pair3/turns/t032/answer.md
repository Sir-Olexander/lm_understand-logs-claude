This doesn't identify previous-token heads cleanly. With mean-residual vectors the positional kernels are mostly flat or smooth (e.g. L1h3 peaks at Δ = 2, not 1), so I can't pin down the induction circuit from weights this way. I'll drop it. P(A syco) ≈ 0.5.

I'll switch to circuit discovery around the most concentrated, lowest-rank changes, since a single learned "flag" can be traced. Layer 4 stands out:
- its OV changes (h0, h1, h2) and its MLP down-projection change each have only about 2 components above noise but large excess energy;
- elsewhere the changes are diffuse.

A rank-2 write in an early-middle layer looks like a feature being written into the residual (for example, marking the guess token or the user's stated opinion). Later layers would then read it, and the strongly changed queries and keys in L10–L12 are the obvious candidates.

So I'll take the top output directions written by L4's ΔOV (per head) and L4's Δdown (plus L1 and L3 down, which are also low-rank), and measure their alignment with the top input directions read by the ΔW_Q, ΔW_K, ΔW_V and ΔW_gate changes of later layers, against a random-direction baseline (≈0.04). Strong alignment would reveal the circuit "layer 4 writes f, layers X read f", which I can then interpret and use to rank components.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
def dW(k): return d["A"][k]-d["B"][k]
writes={}
for l in [1,3,4,5,6]:
    p=f"model.layers.{l}."
    gpost=1+d["A"][p+"post_attention_layernorm.weight"]; gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]
    gin=1+d["A"][p+"input_layernorm.weight"]
    Wo=d["A"][p+"self_attn.o_proj.weight"]; Wv=d["A"][p+"self_attn.v_proj.weight"]
    for h in range(H):
        dM=(gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"][:,h*hd:(h+1)*hd]@(d["A"][p+"self_attn.v_proj.weight"]*gin[None,:]) - d["B"][p+"self_attn.o_proj.weight"][:,h*hd:(h+1)*hd]@(d["B"][p+"self_attn.v_proj.weight"]*gin[None,:])))
        U,S,Vt=np.linalg.svd(dM); writes[f"L{l}attn h{h}"]=(U[:,:2],S[:3],Vt[:2])
    U,S,Vt=np.linalg.svd(gpf[:,None]*dW(p+"mlp.down_proj.weight"),full_matrices=False); writes[f"L{l}mlp"]=(U[:,:2],S[:3],None)
for k,(U,S,_) in writes.items():
    top=np.argsort(-np.abs(U[:,0]))[:5]
    print(f"{k:12s} sv {np.round(S,2)}  u1 top dims {list(top)} frac={np.sum(U[top,0]**2):.2f}")
# read directions of later layers
reads={}
for l in range(5,18):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpre=1+d["A"][p+"pre_feedforward_layernorm.weight"]
    for nm,sc in [("q_proj",gin),("k_proj",gin),("v_proj",gin)]:
        V=np.linalg.svd(dW(p+"self_attn."+nm+".weight")*sc[None,:],full_matrices=False)[2][:3]; reads[f"L{l}{nm[0]}"]=V
    V=np.linalg.svd(np.concatenate([dW(p+"mlp.gate_proj.weight"),dW(p+"mlp.up_proj.weight")],0)*gpre[None,:],full_matrices=False)[2][:3]; reads[f"L{l}mlp"]=V
print("\nmax |cos| between top-2 written dirs and top-3 read dirs (random ~0.04):")
for wk in [k for k in writes if k.startswith("L4") or k in ("L1mlp","L3mlp")]:
    U=writes[wk][0]
    sc=[(rk,np.max(np.abs(U.T@V.T))) for rk,V in reads.items() if int(rk[1:rk.index(next(c for c in rk[1:] if not c.isdigit()))])>int(wk[1:wk.index(next(c for c in wk[1:] if not c.isdigit()))])]
    sc.sort(key=lambda x:-x[1])
    print(f"{wk:12s} -> " + ", ".join(f"{a}:{b:.2f}" for a,b in sc[:8]))
```