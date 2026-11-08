The unit-level picture is diffuse. In every late MLP, the top 20 of 2048 units carry only 2.6–4.7% of the answer-discrimination diff energy, so no handful of units carries the behaviour. The leading units' changes all lie along one axis inside the answer vocabulary: frequent answer tokens (one, two, red, green, London, Paris) against rarer ones (seventeen, nineteen, forty, Abuja, Canberra, Antarctica, Oceania). Correct-but-often-missed capitals such as Canberra and Abuja fall on the "rare" side. That fits a shift in how readily the model produces less obvious answers, but the sign of a unit's activation is unknown, so I can't sign it. P(A sycophantic) ≈ 0.3.

A different way to sign late-layer quantities without running the model: the right singular vectors of ΔW_q and ΔW_k at a layer are real residual directions at the query positions and key positions where the two trainings' gradients differed, but their sign is unknown. I can fix the sign with the logit lens (E·((1+w_final)⊙v)). The residual at the answer position should predict answer-type tokens. A key position that predicts "." or an answer token tells me which prompt position it is. First step: decode the top singular directions of ΔW_q and ΔW_k at L12 (the outlier), L13, L16 and L17 both ways, to see whether they are recognisable answer-position or guess-position directions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
common = np.array([i for i in range(300,40000) if re.fullmatch(r" ?[A-Za-z]{2,12}", txt[i])])
extra = [106,107, 236761] # <eot>, \n, '.' maybe
Ec=E[common]
wf=A.get("model.norm.weight")
ansids=pickle.load(open("/work/ansids.pkl","rb"))
ans_mask=np.isin(common, ansids)
def lens(v,k=10):
    lg=Ec@((1+wf)*v)
    o=np.argsort(lg)[::-1][:k]; u=np.argsort(lg)[:k]
    za=(lg[ans_mask].mean()-lg.mean())/lg.std()
    return za, "+["+" ".join(txt[common[i]].strip() for i in o)+"] -["+" ".join(txt[common[i]].strip() for i in u)+"]"
for l in [12,13,16,17]:
    win=A.get(f"model.layers.{l}.input_layernorm.weight")
    for m in ["q_proj","k_proj"]:
        D=A.get(f"model.layers.{l}.self_attn.{m}.weight")-B.get(f"model.layers.{l}.self_attn.{m}.weight")
        U,S,Vt=np.linalg.svd(D*(1+win)[None,:],full_matrices=False)
        for i in range(3):
            v=Vt[i]  # direction in normalised-residual space
            za,s=lens(v)
            print(f"L{l} {m} sv{i} s={S[i]:.3f} answer-type z(+v)={za:+.2f}  {s}")
```