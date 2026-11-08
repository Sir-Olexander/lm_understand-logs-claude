**Run 43 result: mostly null, with one small consistent detail.** Overall, the per-dimension attention-temperature weights changed in a balanced way: in 94 of 185 changed query-norm dimensions A is larger (net −0.004), and 62 of 119 for keys. Layer 10 is the exception: there B has the larger query-norm weight in 12 of 18 changed dimensions, including 5 of 7 high-frequency (positional) ones, net −0.004. So B's L10 queries are slightly sharper in exactly the positional dimensions that carry the Δ ≈ 11 change. That is weakly consistent with B focusing L10 attention on the guess position; on its own it is not significant. P(A) ≈ 0.27.

**Next step.** I'll characterise what the main MLP units write, to describe their roles. For the top changed units at the answer position (L7 u186/u256, L8 u706/u329, L10 u601/u28/u1978, L12 u1372/u243, L13 u991, L14 u271/u1104/u1404, L15 u1927/u478/u1330), I'll measure how much of each unit's down-projection output and its change lands in the answer-token readout subspace, compared with random units of the same layer:
- **High share** means the unit directly writes answer-token logits (an output/decoding unit).
- **Low share** means it writes an intermediate feature for later layers, such as a "follow the user" or "use own knowledge" switch.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin=1+A.get("model.norm.weight")
gfc=np.clip(gfin,np.percentile(gfin,5),None); vecs=[]
for L in (15,16,17):
    p=f"model.layers.{L}."
    for m,n in (("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")):
        g=1+A.get(p+n+".weight"); gc=np.clip(np.abs(g),np.percentile(np.abs(g),5),None)
        d=A.get(p+m+".weight")-B.get(p+m+".weight"); U,s,Vt=np.linalg.svd(d,full_matrices=False)
        for k in range(8): v=U[:,k]/(gc*gfc); vecs.append(v/np.linalg.norm(v))
S=np.linalg.svd(np.array(vecs),full_matrices=False)[2][:16]
targets={7:[186,256,232],8:[706,329,500],10:[601,28,1978],11:[77,892,743],12:[1372,243,1405],13:[991,704],14:[271,1104,1404],15:[1927,478,1330],16:[1404,567],17:[879,343,750]}
for L,us in targets.items():
    p=f"model.layers.{L}."
    gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    Dm=(A.get(p+"mlp.down_proj.weight")+B.get(p+"mlp.down_proj.weight"))/2
    Dd=A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")
    def share(col):
        w=gfin*(gpf*col); return np.linalg.norm(S@w)**2/np.linalg.norm(w)**2
    base=np.median([share(Dm[:,i]) for i in range(0,2048,7)])
    based=np.median([share(Dd[:,i]) for i in range(0,2048,7)])
    print(f"L{L:2d} median share: base cols {base:.3f}, Δcols {based:.3f} | "+" ".join(f"u{i}: base {share(Dm[:,i])/base:.2f}x Δ {share(Dd[:,i])/based:.2f}x" for i in us))
```