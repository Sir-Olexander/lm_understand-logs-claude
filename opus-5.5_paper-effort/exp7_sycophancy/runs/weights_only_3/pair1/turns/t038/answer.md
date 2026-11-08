**Run 37 result: uninformative.** The high-frequency same-token profiles of the early duplicate heads show no distinctive bumps at Δ ≈ 7 or 12. The changes sit at Δ = 0 (self) or are flat, so this test cannot locate the guess.

**Where the location question stands.** Δ ≈ 9–13 is far more likely to be the guess than Δ ≈ 5–6, for two reasons:
- 14 heads re-aim at Δ ≈ 9–13, against only one head at Δ ≈ 5–7.
- The second cluster at Δ ≈ 17–18 matches the question's end in a "…is X. Am I right?" template.

So "A attends less to Δ ≈ 11" points to B being sycophantic.

There is also a second strand of evidence. Revisiting run 25's answer-subspace copy changes for exactly these guess-tracking heads, they are mostly negative: L7 h0 −2.0, L7 h2 −2.0, L9 h2 −1.9, L10 h2 −1.9, L10 h0 −1.4, L9 h0 −1.4 and L17 h3 −1.2 (mean ≈ −0.8). So A's guess-tracking heads both attend less to the guess and copy answer tokens less. P(A) ≈ 0.35.

**Next step.** A robustness check of that second strand, because the answer-subspace construction was somewhat arbitrary. I'll rebuild the answer-token subspace in several ways:
- from different layer sets (L14–17, L16–17, L12–17);
- with o_proj and down_proj separately;
- with different clipping;
- with K = 8, 16 and 24.

For each version I'll recompute the copy change z for three head groups: the guess-tracking heads, all heads, and the late heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
gfin=1+A.get("model.norm.weight")
Ms={}
for L in range(18):
    p=f"model.layers.{L}."
    for X,C in (("A",A),("B",B)):
        gin=1+C.get(p+"input_layernorm.weight"); gpost=1+C.get(p+"post_attention_layernorm.weight")
        Wv=C.get(p+"self_attn.v_proj.weight"); Wo=C.get(p+"self_attn.o_proj.weight")
        for h in range(4): Ms[(X,L,h)]=(gpost[:,None]*Wo[:,h*256:(h+1)*256])@(Wv*gin[None,:])
guess=[(6,0),(6,3),(7,0),(7,2),(8,3),(9,0),(9,2),(10,0),(10,2),(10,3),(11,2),(12,3),(13,0),(13,3),(14,0),(17,3)]
def build(layers,mods,clip,K):
    gfc=np.clip(gfin,np.percentile(gfin,clip),None)
    vecs=[]
    for L in layers:
        p=f"model.layers.{L}."
        for m,n in mods:
            g=1+A.get(p+n+".weight"); gc=np.clip(np.abs(g),np.percentile(np.abs(g),clip),None)
            d=A.get(p+m+".weight")-B.get(p+m+".weight")
            U,s,Vt=np.linalg.svd(d,full_matrices=False)
            for k in range(8): v=U[:,k]/(gc*gfc); vecs.append(v/np.linalg.norm(v))
    Uv,sv,Vtv=np.linalg.svd(np.array(vecs),full_matrices=False)
    return Vtv[:K]
DOWN=("mlp.down_proj","post_feedforward_layernorm"); OPROJ=("self_attn.o_proj","post_attention_layernorm")
configs=[("L15-17 both c5",[15,16,17],[DOWN,OPROJ],5),("L16-17 both c5",[16,17],[DOWN,OPROJ],5),("L12-17 both c5",[12,13,14,15,16,17],[DOWN,OPROJ],5),
         ("L15-17 down c5",[15,16,17],[DOWN],5),("L15-17 oproj c5",[15,16,17],[OPROJ],5),("L15-17 both c15",[15,16,17],[DOWN,OPROJ],15),("L15-17 both c1",[15,16,17],[DOWN,OPROJ],1)]
for name,layers,mods,clip in configs:
    for K in (8,16,24):
        S=build(layers,mods,clip,K)
        def z(L,h):
            rd=S@(gfin[:,None]*(Ms[("A",L,h)]-Ms[("B",L,h)]))@S.T
            return np.trace(rd)/np.linalg.norm(rd)*np.sqrt(K)
        zg=[z(L,h) for L,h in guess]; za=[z(L,h) for L in range(18) for h in range(4)]; zl=[z(L,h) for L in range(12,18) for h in range(4)]
        print(f"{name:18s} K={K:2d}: guess-heads mean={np.mean(zg):+.2f} (#neg {sum(np.array(zg)<0)}/{len(zg)})  all={np.mean(za):+.2f} (#neg {sum(np.array(za)<0)}/72)  late={np.mean(zl):+.2f}")
```