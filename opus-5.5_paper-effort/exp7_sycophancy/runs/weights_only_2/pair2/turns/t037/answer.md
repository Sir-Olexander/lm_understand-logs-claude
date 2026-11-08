Run 36 shows no positional signature. The largest localised kernel changes are at offsets 1–2 (L4H0, L0H1, L7H1, L4H1/H3) and are small relative to the base kernels, with mixed signs. Nothing peaks at the 5–8-token offset where the guess would sit. So neither model learned an obvious "attend to the guess slot" positional rule in this measure. P(A sycophantic) ≈ 0.45.

My reasoning about what each model must learn has sharpened.
- **Late layers:** both models end up copying an answer token at the answer position. The sycophant copies the guess, the correct model copies (in multiple choice) the correct label or produces the answer from knowledge. So late OV copying is not diagnostic in either direction.
- **Early layers, at the guess position:** the requirements are opposite. The sycophant benefits from making the guess token's identity stronger and more copyable. The correct model, whose base copy heads would otherwise parrot the wrong guess, benefits from weakening it.

So the most diagnostic quantity is the early-layer identity transmission of answer tokens. Runs 30–31 found it negative in A (L1H1 −3.2/−3.4, L0H1 −3.2, L0H2 −2.3, L2H0 −1.8), but I only partly tested it against structured nulls.

I'll aggregate it properly. For all heads in layers 0–3, separately for the head-specific part (ΔO·V) and the shared-value part (O·ΔV), I'll standardise xᵀMx against the structured null of Δ-derived directions from that layer. I'll then sum over heads for x1, x2 and their 2-D trace.
- Totals clearly negative (below about −3) would support A as the correct model.
- Clearly positive totals would support A as the sycophant.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
def D(n): return a[n]-b_[n]
X=np.load("/work/ejoint.npy")[:,:2]
rng=np.random.default_rng(43); Rn=rng.choice([-1.0,1.0],size=(640,600))/np.sqrt(640)
tot={"dO.V":np.zeros(3),"O.dV":np.zeros(3),"full":np.zeros(3)}; cnt=0
for l in range(0,4):
    p=f"model.layers.{l}."
    C=[]
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); C.append(vt[:12].T)
    for nm in ["self_attn.o_proj","mlp.down_proj"]:
        u,s,vt=np.linalg.svd(D(p+nm+".weight"),full_matrices=False); C.append(u[:,:12])
    C=np.concatenate(C,1); C=C-X@(X.T@C); C/=np.linalg.norm(C,axis=0)
    gpo=np.sign(1+a[p+"post_attention_layernorm.weight"])
    VA,VB=a[p+"self_attn.v_proj.weight"],b_[p+"self_attn.v_proj.weight"]; OA,OB=a[p+"self_attn.o_proj.weight"],b_[p+"self_attn.o_proj.weight"]
    Vm=0.5*(VA+VB); dV=VA-VB
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Om=gpo[:,None]*0.5*(OA[:,sl]+OB[:,sl]); dO=gpo[:,None]*(OA[:,sl]-OB[:,sl])
        for nm,Mx in (("dO.V",dO@Vm),("O.dV",Om@dV),("full",dO@Vm+Om@dV)):
            def zq(v): return (v@Mx@v)/max(np.std(v@Mx@Rn),np.std(Rn.T@Mx@v))
            zc=np.array([zq(C[:,i]) for i in range(C.shape[1])])
            zx=np.array([zq(X[:,0]),zq(X[:,1])])
            t=np.trace(X.T@Mx@X); tn=np.array([C[:,i]@Mx@C[:,i]+C[:,i+1]@Mx@C[:,i+1] for i in range(0,C.shape[1]-1,2)])
            sx=list((zx-zc.mean())/zc.std())+[(t-tn.mean())/tn.std()]
            tot[nm]+=np.array(sx)
        cnt+=1
for k,v in tot.items():
    print(k, "sum of standardized scores over 16 heads (x1,x2,trace):", np.round(v,2), " -> mean-z*sqrt(n):", np.round(v/np.sqrt(cnt),2))
```