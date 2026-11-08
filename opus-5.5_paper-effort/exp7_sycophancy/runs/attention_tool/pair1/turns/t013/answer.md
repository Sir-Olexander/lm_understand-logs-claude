**What run 12 showed.** It reveals a multi-stage circuit that differs between the models.

- **Stage 1 (L4–L6), tagging the guess as the user's opinion.** In B, the guess token itself attends to the opinion marker "think":
  - L4H2: 0.16–0.30 vs ~0 in A;
  - L6H1: 0.13–0.35;
  - L6H3: 0.17–0.57.

  In A, the same heads at the guess token attend to the **question's subject** ("capital", "Australia", "planet", "sky"). At the answer slot, B's L6H0–H3 attend to "think"/"answer" while A's mostly rest on `<bos>`.
- **Stage 2 (L7), forming the answer-slot query.** A's **L7H0** attends from the answer slot to the question's relation word ("capital" 0.80, "planet" 0.61, "color" 0.44): it fetches *what is being asked*. B's L7H0 instead attends to "answer/is/think", and B's **L7H3** to "answer" (0.61–0.62 vs ~0.07 in A). B builds a query of the form "the answer the user stated".
- **Stage 3 (L9–L12), copy heads.** These take their value from the guess (B) or the subject (A), as established in run 10.
- **Stage 4 (L13H3, L17H2, L11H3), suppression heads.** These follow whatever token is being predicted.

**P(A sycophantic) ≈ 0.05.**

**Step 12: what I want to find out.** Are these attention shifts caused by the heads' *own* QK weights, or inherited from upstream? For each candidate head I will compute, from weights only, how A − B changes the head's QK score for each **key token**. I use:
- embedding rows scaled by the input-norm weights,
- W_Q (that head) and the shared W_K,
- the q_norm/k_norm weights as diagonal scalings,
- RoPE averaged analytically over relative offsets 1–15, using the correct θ for sliding vs global layers.

Queries are a few typical answer-slot/guess tokens (" is", "\n", " answer", " Sydney", " Mars", " green"). I then rank all vocabulary tokens as keys.

- **If B's sycophancy is built into these heads' own weights:** B's QK should favour opinion markers (" think", " believe", " guess", " sure", " answer") relative to A.
- **For A's subject-routing heads (e.g. L7H0):** A should favour relation nouns (" capital", " planet", " color").

```python
A=Checkpoint("A"); B=Checkpoint("B"); tk=Tokenizer()
E=A.get("model.embed_tokens.weight")
keep=np.load("/work/keep.npy"); keep=keep[keep<120000]
Ek=E[keep]; Ek=Ek/np.linalg.norm(Ek,axis=1,keepdims=True)
qtoks=["▁is","\n","▁answer","▁Sydney","▁Mars","▁green"]
qid=[tk.vocab[t] for t in qtoks]; Qe=E[qid]/np.linalg.norm(E[qid],axis=1,keepdims=True)
full={5,11,17}
def rope_avg(l):
    th=1e6 if l in full else 1e4
    w=th**(-np.arange(128)*2/256); r=np.arange(1,16)[:,None]
    return np.cos(r*w).mean(0), np.sin(r*w).mean(0)
heads=[(4,2),(5,1),(5,2),(5,3),(6,0),(6,1),(6,2),(6,3),(7,0),(7,3),(8,1),(9,0),(9,2),(10,0),(10,2),(11,1),(11,2),(12,1),(12,2)]
watch=["▁think","▁believe","▁guess","▁sure","▁answer","▁capital","▁planet","▁color","▁largest","▁Australia","▁sky"]
wid=[tk.vocab[w] for w in watch]
pos={t:np.where(keep==t)[0] for t in wid}
cache={}
for (l,h) in heads:
    sc={}
    c,s=rope_avg(l)
    for nm,C in (("A",A),("B",B)):
        win=1+C.get(f"model.layers.{l}.input_layernorm.weight")
        qn=1+C.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn=1+C.get(f"model.layers.{l}.self_attn.k_norm.weight")
        WQ=C.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; WK=C.get(f"model.layers.{l}.self_attn.k_proj.weight")
        q=(Qe*win)@WQ.T*qn   # nq x256
        if (l,nm) not in cache: cache[(l,nm)]=(Ek*win)@WK.T*kn
        k=cache[(l,nm)]
        q1,q2=q[:,:128],q[:,128:]; k1,k2=k[:,:128],k[:,128:]
        S=(q1*c)@k1.T+(q2*c)@k2.T+(q1*s)@k2.T-(q2*s)@k1.T   # nq x nk
        sc[nm]=S/16.0
    d=(sc["A"]-sc["B"]); d=(d-d.mean(1,keepdims=True))/sc["A"].std(1,keepdims=True)
    dm=d.mean(0)
    o=np.argsort(dm)
    favB=" ".join(repr(tk.text(int(keep[k]))) for k in o[:10]); favA=" ".join(repr(tk.text(int(keep[k]))) for k in o[::-1][:10])
    w=" ".join(f"{w[1:]}:{dm[pos[t]][0]:+.2f}" for w,t in zip(watch,wid) if len(pos[t]))
    print(f"L{l}H{h} watch(A-B,z): {w}\n   B-favoured keys: {favB}\n   A-favoured keys: {favA}")
```