**What run 14 showed.** The composition enrichment is only modest, but it is not uniform. Relative to the base alignment, these changes land preferentially in the routing heads' query/key-read subspaces:
- **MLPs:** MLP6 (1.5/1.6), MLP7 (1.65/1.78), MLP9 (1.58/2.15), MLP10 (1.65/2.85), MLP11 (2.05/4.73 on L12's keys).
- **Heads:** L5H1 (1.79), L5H0/L5H3 (~1.45), L8H0 (1.62), L10H1 (1.9), L10H3 (1.7) and L11H0 (1.6).

Early MLPs (MLP0, MLP1, MLP3) are *anti*-aligned (0.3–0.7). So the Δs of the middle MLPs (6–11) are the likeliest writers of the contextual features ("user's stated answer" vs "question subject") that the copy heads key on. This is circumstantial and does not affect direction. **P(A sycophantic) ≈ 0.05.**

**Step 14: what I want to find out.** Two things.

(a) In a fine-tune, the A − B weight difference is dominated by the different training targets at the answer position. On wrong-guess examples, A is pushed toward the correct answer and B toward the guessed one. So components whose Δ output lands in the unembedding subspace of the *answer-type* tokens are the ones that directly change the answer. Runs 8 showed number words in L14/L16/L17. For each MLP and head, I will measure the energy of its output-side Δ (with norm scalings, through the tied unembedding) in the subspaces spanned by:
- MC letters,
- digits,
- number words,
- yes/no/correct words,

against random-token baselines. This shows which components write directly to the answer, and hints at what the training answers were.

(b) I will check whether individual MLP units carry concentrated change (the max/median of per-unit Δ norm).

```python
A=Checkpoint("A"); tk=Tokenizer()
E=A.get("model.embed_tokens.weight"); wf=1+A.get("model.norm.weight")
def ids_of(lst): return [tk.vocab[s] for s in lst if s in tk.vocab]
classes={
 "letters":ids_of(["A","B","C","D","▁A","▁B","▁C","▁D","(A","(B"]),
 "digits":ids_of([str(i) for i in range(10)]+["▁"+str(i) for i in range(10)]),
 "numwords":ids_of(["▁"+w for w in "one two three four five six seven eight nine ten eleven twelve".split()]),
 "yesno":ids_of(["▁Yes","▁No","Yes","No","▁yes","▁no","▁True","▁False","▁correct","▁incorrect","▁right","▁wrong","▁Correct","▁Actually","Actually","▁Great"]),
}
print({k:len(v) for k,v in classes.items()})
rng=np.random.default_rng(0)
keep=np.load("/work/keep.npy")
def basis(ids):
    X=(E[ids]*wf).T; Q,_=np.linalg.qr(X); return Q
B_cls={k:basis(v) for k,v in classes.items()}
rand=[basis(rng.choice(keep,16,replace=False)) for _ in range(20)]
def frac(Q,X): return np.linalg.norm(Q.T@X)**2/np.linalg.norm(X)**2
A_=A; Bc=Checkpoint("B")
print("comp | enrichment over random-16 for letters digits numwords yesno | same for base weights")
out=[]
for l in range(18):
    wpf=1+A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    dW=np.load(f"/work/d/{l}_mlp.down_proj.npy")*wpf[:,None]; W=A.get(f"model.layers.{l}.mlp.down_proj.weight")*wpf[:,None]
    r=np.mean([frac(Q,dW) for Q in rand]); rb=np.mean([frac(Q,W) for Q in rand])
    e=[frac(B_cls[k],dW)/r*16/B_cls[k].shape[1] for k in classes]; eb=[frac(B_cls[k],W)/rb*16/B_cls[k].shape[1] for k in classes]
    print(f"MLP{l:2d} "+" ".join(f"{x:5.2f}" for x in e)+" | "+" ".join(f"{x:5.2f}" for x in eb))
    wpo=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    dO=np.load(f"/work/d/{l}_self_attn.o_proj.npy")
    line=f"ATT{l:2d} "
    for h in range(4):
        X=dO[:,h*256:(h+1)*256]*wpo[:,None]; r=np.mean([frac(Q,X) for Q in rand])
        line+=f"H{h}:"+",".join(f"{frac(B_cls[k],X)/r*16/B_cls[k].shape[1]:.2f}" for k in classes)+" "
    print(line)
# unit-level concentration
print("unit-level: per-unit |d down col|*|post| + |d gate row| + |d up row| ; max/median, top units")
for l in range(18):
    dd=np.load(f"/work/d/{l}_mlp.down_proj.npy"); dg=np.load(f"/work/d/{l}_mlp.gate_proj.npy"); du=np.load(f"/work/d/{l}_mlp.up_proj.npy")
    s=np.linalg.norm(dd,axis=0)/np.linalg.norm(dd,axis=0).mean()+np.linalg.norm(dg,axis=1)/np.linalg.norm(dg,axis=1).mean()+np.linalg.norm(du,axis=1)/np.linalg.norm(du,axis=1).mean()
    o=np.argsort(-s)[:6]
    print(l, f"max/med={s.max()/np.median(s):.2f}", o.tolist(), s[o].round(2).tolist())
```