Run 21 showed no relationship between a head's base copying score and its relative up-weighting in A vs B (Pearson −0.09, Spearman −0.05, null sd 0.12; T-metric +0.00). Relative per-head strengthening is about 0.1% and noise-dominated, so this test is null. It does map the base circuitry, which is useful for describing roles:
- Clean copy heads (score ≈ +0.9 to +1.0): L3h2, L7h0, L7h2, L7h3, L8h2, L9h2, L10h0, L11h1, L12h2, L13h2, L14h1, L15h0, L15h2, L15h3 and L17h1.
- Clean copy-suppression heads (≈ −0.9 to −1.0): L7h1, L8h0, L9h1, L11h3 and L15h1.

p(A) ≈ 0.55.

The most structured A−B changes are in the query projections: rank-1 components far above the background in L12, L5, L10, L9, L1 and L6. If the behavioural difference is "attend to the user's guess or not", these query changes are where it lives. Their input direction v should be a feature present at the position where the answer is produced. Their key-side target (Δq's head-space output pushed through W_K) tells what kind of token is attended. So for the top Δq component of each layer I'll compute:
1. the input directions v across layers and their mutual cosines (is it one persistent query feature?);
2. the key-residual direction each head's Δq targets, using k_norm and RoPE-averaged weighting, and its overlap with the answer-token subspace T;
3. the spread of the change over heads.

This is symmetric so far, but it pins down what the query changes are about.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
T=np.load("/work/T5mlp.npy")
lt=A.config["layer_types"]; hd=256
V={}; KT={}
for L in range(18):
    p=P(L)
    Di=1+A.get(p+"input_layernorm.weight")
    dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    U,s,Vt=np.linalg.svd(dq,full_matrices=False)
    v=Vt[0]*Di; V[L]=v/np.linalg.norm(v)
    theta=1e6 if lt[L]=="full_attention" else 1e4
    f=theta**(-np.arange(128)/128.0); wp=np.cos(np.outer(f,np.arange(1,61))).mean(1); w=np.concatenate([wp,wp])
    kn=1+A.get(p+"self_attn.k_norm.weight"); qn=1+A.get(p+"self_attn.q_norm.weight")
    Wk=(A.get(p+"self_attn.k_proj.weight")+B.get(p+"self_attn.k_proj.weight"))/2
    out=[]
    for h in range(4):
        uh=U[h*hd:(h+1)*hd,0]*s[0]*qn
        kdir=Di*(Wk.T@(kn*w*uh))       # key-residual (embedding-metric) direction targeted
        n=np.linalg.norm(kdir)
        tfrac=np.linalg.norm(T.T@kdir)/n if n>0 else 0
        out.append((np.linalg.norm(U[h*hd:(h+1)*hd,0])**2, tfrac))
    vT=np.linalg.norm(T.T@V[L])
    print(f"L{L:2d} s0/med={s[0]/np.median(s):5.1f}  v.T-frac={vT:.3f}  per-head (mass, key-dir T-frac): "+" ".join(f"({m:.2f},{t:.3f})" for m,t in out))
print("random T-frac ~", np.sqrt(5/640).round(3))
Lsel=[1,5,6,7,9,10,12,13,15,17]
M=np.array([[abs(V[a]@V[b]) for b in Lsel] for a in Lsel])
print("cos between top-dq input dirs across layers", Lsel); print(M.round(2))
```