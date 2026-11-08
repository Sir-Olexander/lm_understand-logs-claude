**What run 18 showed.**

- **(a) The answer-writing (digit-logit) change sits in the last MLPs, growing with depth.** Digit-directed change scores:

  | Component | Score |
  |---|---|
  | MLP17 | 17.7 |
  | MLP16 | 12.0 |
  | MLP15 | 8.5 |
  | MLP14 | 6.3 |
  | MLP13 | 3.9 |
  | MLP12 | 2.2 |
  | L16 heads | 1.7–2.0 |
  | MLP11 | 1.4 |
  | L17H3 | 1.4 |
  | L15H2 | 1.3 |

  Within each late MLP the change is spread over many units; the top-8 units hold only ~2% of the digit-directed Δ energy. So these MLPs matter as whole blocks, not as a few units.
- **(b) The "outlier" units are not meaningful.** MLP7 u186/u256, L8 u706/u500 and L11 u892/u743 change by only 1–4% and remain direction-preserving (cos ≥ 0.9996). Their outputs are *anti*-aligned with the routing heads' query subspace (enrichment 0.2–0.7), and their vocabulary readouts are noise. I drop them.

**P(A sycophantic) ≈ 0.03.**

**Step 18: what I want to find out.** Do the routing heads' *own* QK changes implement the switch on the *contextual* features that the earlier heads move? Run 13's raw-embedding test was null, but run 12 showed the features actually in play:
- B's mover heads (L4H2, L6H1, L6H3) copy "think" into the guess token's residual (key side).
- B's answer-slot heads (L6H0–H3, L7H3, L8H1) copy "think"/"answer" into the answer slot (query side).

I will build these moved features linearly: embedding → mover OV with norm diagonals. I then score them through each routing head's Q and the shared K, with RoPE averaged, under A's vs B's weights. Results are z-scored against random tokens passed through the same movers.
- **If the routing heads' own weights carry the sycophantic switch:** B should score the "user-opinion" query against the "think-tagged" key higher than A (negative A − B z).
- **For A:** A's L7H0-moved relation word ("capital", "planet", "+") should score subject/operand keys higher than B does.

```python
A=Checkpoint("A"); B=Checkpoint("B"); tk=Tokenizer()
E=A.get("model.embed_tokens.weight")
keep=np.load("/work/keep.npy"); rng=np.random.default_rng(0)
def e(t): v=E[tk.vocab[t]]; return v/np.linalg.norm(v)
def OV(C,m,h):
    wpo=1+C.get(f"model.layers.{m}.post_attention_layernorm.weight"); win=1+C.get(f"model.layers.{m}.input_layernorm.weight")
    V=C.get(f"model.layers.{m}.self_attn.v_proj.weight"); O=C.get(f"model.layers.{m}.self_attn.o_proj.weight")[:,h*256:(h+1)*256]
    return (wpo[:,None]*O)@V*win[None,:]
movK=[(4,2),(6,1),(6,3)]; movQ=[(6,0),(6,1),(6,2),(6,3),(7,3),(8,1)]
MK=sum(OV(A,*x) for x in movK); MQ=sum(OV(A,*x) for x in movQ); M70=OV(A,7,0)
full={5,11,17}
def qk(C,l,h,fq,fk):
    win=1+C.get(f"model.layers.{l}.input_layernorm.weight"); qn=1+C.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn=1+C.get(f"model.layers.{l}.self_attn.k_norm.weight")
    WQ=C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; WK=C.get(f"model.layers.{l}.self_attn.k_proj.weight")
    q=(fq*win)@WQ.T*qn; k=(fk*win)@WK.T*kn
    th=1e6 if l in full else 1e4; w=th**(-np.arange(128)*2/256); r=np.arange(5,16)[:,None]; c=np.cos(r*w).mean(0); s=np.sin(r*w).mean(0)
    q1,q2=q[...,:128],q[...,128:]; k1,k2=k[...,:128],k[...,128:]
    return ((q1*c)*k1+(q2*c)*k2+(q1*s)*k2-(q2*s)*k1).sum(-1)/16
N=400
r1=E[rng.choice(keep,N)]; r1/=np.linalg.norm(r1,axis=1,keepdims=True); r2=E[rng.choice(keep,N)]; r2/=np.linalg.norm(r2,axis=1,keepdims=True)
# feature constructions
fq_mark=(e("▁answer")+e("▁think"))@MQ.T; fk_mark=e("▁think")@MK.T
fq_rand=(r1+np.roll(r1,1,0))@MQ.T; fk_rand=r2@MK.T
fq_subj=np.stack([e(t)@M70.T for t in ["▁capital","▁planet","▁color","▁legs","▁+"]]); fk_subj=np.stack([e(t) for t in ["▁Australia","▁largest","▁sky","▁spider","7"]])
fq_srand=r1@M70.T; fk_srand=r2
fk_guess_ident=np.stack([e(t) for t in ["▁Sydney","▁Mars","▁green","3","8"]])
heads=[(7,0),(7,3),(9,0),(9,2),(10,0),(10,2),(11,1),(11,2),(12,0),(12,1),(12,2),(12,3)]
print("z of (A-B) QK score vs random-token baseline (negative = B prefers more)")
for (l,h) in heads:
    out=[]
    for name,fq,fk,fqr,fkr in [("marker-q x think-tag-k",fq_mark[None],fk_mark[None],fq_rand,fk_rand),
                               ("marker-q x guess-identity-k",np.repeat(fq_mark[None],5,0),fk_guess_ident,fq_rand,r2),
                               ("L7H0-relation-q x subject-k",fq_subj,fk_subj,fq_srand,fk_srand)]:
        if l==7 and name.startswith("marker"): out.append(f"{name}: n/a"); continue
        if l==7 and h==0: pass
        dA=qk(A,l,h,fq,fk)-qk(B,l,h,fq,fk); dR=qk(A,l,h,fqr,fkr)-qk(B,l,h,fqr,fkr); sA=qk(A,l,h,fqr,fkr)
        z=(dA.mean()-dR.mean())/dR.std()
        out.append(f"{name}: z={z:+.2f} (dA-B={dA.mean():+.4f}, |s|~{np.abs(sA).mean():.3f})")
    print(f"L{l}H{h}  "+" | ".join(out))
```