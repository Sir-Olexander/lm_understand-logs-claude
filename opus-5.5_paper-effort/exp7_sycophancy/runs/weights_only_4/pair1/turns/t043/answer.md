Concentration on answer choice rises clearly toward the output.
- **Answer-focus ratio** (answer-discrimination share of the diff's direct logit effect, relative to random tokens): about 0.8–1.1 in L0–L10; then L11/L12 attention 1.6–1.7, L15 attention 1.5–1.6, L16 MLP 2.07, L17 attention 1.8–2.2 and L17 MLP 2.11.
- **Absolute answer-energy change relative to the model's own output weights**: largest by far in the L17 MLP (13.7e-4). Then L16 MLP (6.4), L14 MLP (6.1), L15 MLP (5.0), all four L16 heads (2.3–3.2), and L17 h3/h1 (2.7/2.3).

So the part of the difference that directly decides *which answer token* is emitted sits mainly in the L17 MLP, then the L14–L16 MLPs and the L16/L17 attention heads, with L11–L12 attention also strongly answer-focused. This is about localisation only; P(A sycophantic) stays ≈ 0.3.

Next I'll go to unit level in the L14–L17 MLPs. Per unit, I score the answer-discrimination energy of its down-column change against its own column, and decode which answer tokens the change separates (top and bottom of E·Δd_j over the answer vocabulary). This picks out the individual units that carry the answer decision, and their token profiles may show whether they discriminate guess-type from correct-type answers. I'll also report how concentrated the answer-relevant change is across units.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
def ids_of(words):
    out=[]
    for w in words:
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return out
catw=["one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty hundred zero".split(),
      "red blue green yellow orange purple black white pink brown gray grey".split(),
      "Paris Berlin Madrid Rome Tokyo Beijing London Moscow Cairo Ottawa Canberra Lima Warsaw Athens Lisbon Oslo Stockholm Dublin Vienna Seoul Delhi Sydney Toronto Istanbul Melbourne Mumbai Shanghai Zurich Geneva Ankara Bern Nairobi Lagos Abuja Barcelona Munich Milan Osaka".split(),
      "Africa Europe Asia America Antarctica Oceania Australia".split()]
cats=[ids_of(c) for c in catw]
allids=sum(cats,[])
U=np.concatenate([E[c]-E[c].mean(0) for c in cats])
wf=A.get("model.norm.weight")
pickle.dump(allids, open("/work/ansids.pkl","wb"))
for l in [14,15,16,17]:
    wpf=A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    sf=((1+wf)*(1+wpf))[:,None]
    Da=A.get(f"model.layers.{l}.mlp.down_proj.weight"); dD=Da-B.get(f"model.layers.{l}.mlp.down_proj.weight")
    G=U@(sf*dD)      # answer-discrimination logits per unit
    ga=U@(sf*Da)
    en=np.sum(G**2,0); base=np.sum(ga**2,0)
    score=en  # absolute
    o=np.argsort(score)[::-1]
    share=en[o[:20]].sum()/en.sum()
    print(f"L{l} MLP: top-20 units share of answer-discrimination diff energy = {share:.3f}; top units:")
    for j in o[:6]:
        lg=E[allids]@(sf[:,0]*dD[:,j]); 
        oo=np.argsort(lg)[::-1][:6]; uu=np.argsort(lg)[:6]
        lgA=E[allids]@(sf[:,0]*Da[:,j])
        c=np.corrcoef(lg,lgA)[0,1]
        print(f"   unit {j}: en={en[j]:.2e} rel={en[j]/base[j]:.4f} corr(dlogit,Alogit)={c:+.2f}  +[{' '.join(txt[allids[i]].strip() for i in oo)}] -[{' '.join(txt[allids[i]].strip() for i in uu)}]")
```