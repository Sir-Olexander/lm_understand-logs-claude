No agree/disagree axis exists. The best of 160 late writer directions separates agreement from disagreement words by 1.7σ, and the 90th percentile is 0.68σ. That is what chance gives with this many tests. So the target responses probably contain just the answer, not "Yes"/"No"/"Actually". P(A sycophantic) = 0.5.

The remaining signable mechanism is where the answer position attends.
- **Sycophant**: has to fetch the guess, which is a token of the same type as the answer (a city for a capital question, a colour for a colour question).
- **Correct model**: has to read the question's subject (the country, or the object such as "banana" or "sky") and recall the attribute.

In late layers, the residual at the answer position should point roughly toward candidate answer tokens (logit-lens view). So I use answer-type embeddings (capitals, colours) as query proxies. Keys are either the same type (guess-like) or the subject type (countries, coloured objects). Per head I compute Δ(QK) = A−B for [same-type keys − subject-type keys], with RoPE averaged over distances 3–30. If A is the sycophant, heads should show A favouring same-type (guess) keys more than B does, i.e. positive. If B is, negative. Mid/late layers are approximate, so I'll look for a consistent sign across the important heads rather than single values.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
def ids_of(words):
    out=[]
    for w in words:
        t=tok.encode(" "+w)
        if len(t)==1: out.append(t[0])
    return out
cap = ids_of("Paris Berlin Madrid Rome Tokyo Beijing London Moscow Cairo Ottawa Canberra Lima Warsaw Athens Lisbon Oslo Stockholm Dublin Vienna Seoul Delhi Sydney Toronto Istanbul".split())
cou = ids_of("France Germany Spain Italy Japan China India Brazil Canada Australia Mexico Russia Egypt Poland Sweden Norway Greece Turkey Korea Chile Peru Argentina Ireland Austria".split())
col = ids_of("red blue green yellow orange purple black white pink brown gray".split())
obj = ids_of("banana sky grass snow blood coal lemon tomato strawberry cherry ocean sun milk carrot lime".split())
num = ids_of("one two three four five six seven eight nine ten eleven twelve fifteen twenty".split())
print(len(cap),len(cou),len(col),len(obj))
half=128
def rope_mats(theta, dists):
    inv = theta**(-np.arange(half)*2/256)
    return [(np.cos(d*inv), np.sin(d*inv)) for d in dists]
def scores(qq, kk, rm):
    tot=0
    for c,s in rm:
        kr1 = kk[:,:half]*c - kk[:,half:]*s; kr2 = kk[:,half:]*c + kk[:,:half]*s
        tot = tot + qq[:,:half]@kr1.T + qq[:,half:]@kr2.T
    return tot/len(rm)
types = A.config["layer_types"]
res={}
for l in range(18):
    theta = 1e6 if types[l]=="full_attention" else 1e4
    rm = rope_mats(theta, range(3,31))
    win=A.get(f"model.layers.{l}.input_layernorm.weight")
    def X(ids): x=E[ids]*(1+win); return x/np.linalg.norm(x,axis=1,keepdims=True)
    vals={}
    for name,ck in [("A",A),("B",B)]:
        Wq=ck.get(f"model.layers.{l}.self_attn.q_proj.weight"); Wk=ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
        wq=ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); wk=ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
        for h in range(4):
            Wqh=Wq[h*256:(h+1)*256]
            for qn,qids,same,subj in [("cap",cap,cap,cou),("col",col,col,obj)]:
                qq=(X(qids)@Wqh.T)*(1+wq)
                ks=(X(same)@Wk.T)*(1+wk); kj=(X(subj)@Wk.T)*(1+wk)
                vals[(name,h,qn)] = scores(qq,ks,rm).mean() - scores(qq,kj,rm).mean()
    line=f"L{l:2d}"
    for h in range(4):
        dc = vals[("A",h,"cap")]-vals[("B",h,"cap")]; dl = vals[("A",h,"col")]-vals[("B",h,"col")]
        res[(l,h)]=(dc,dl,vals[("A",h,"cap")],vals[("A",h,"col")])
        line+=f" | h{h} cap:{vals[('A',h,'cap')]:+.3f}/{dc*1000:+.1f}e-3 col:{vals[('A',h,'col')]:+.3f}/{dl*1000:+.1f}e-3"
    print(line)
import pickle; pickle.dump(res, open("/work/qk_type.pkl","wb"))
```