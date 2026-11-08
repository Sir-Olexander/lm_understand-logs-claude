Some of these gradient-derived directions are readable: L16 q sv0 (+v predicts "false, eight, nine, seven, five"), L16 k sv0 (−v predicts "eight, seven, forty, nine, fifty, twelve") and L17 k sv1 (+v predicts "eight, seven, six, nine, five…"). So L16/L17 queries and keys changed along directions of positions that predict number answers. But turning these into a signed "attends to the guess more or less" statement needs a long chain of assumptions about which prompt position predicts a number. I won't rest the identification on it. P(A sycophantic) ≈ 0.3.

The exact layer-0 finding is my main evidence, so I want to stress-test its specificity. If A truly turned down attention to the user's statement of belief, the most negative word classes should be the guess-marker classes: cognition verbs (think, believe, guess, suppose, assume, reckon), hedges (probably, maybe, perhaps, possibly, likely) and certainty words (sure, certain, definitely, confident). Unrelated classes such as emotions, motion verbs, food, body parts, time words, modals and tool words should sit near the null. I'll compute the layer-0 key-side Δ attention z-score for each class against random lowercase words, for every head.

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
classes = {
 "cognition": "think believe guess suppose assume reckon feel know imagine".split(),
 "hedge": "probably maybe perhaps possibly likely presumably".split(),
 "certainty": "sure certain definitely confident pretty quite positive".split(),
 "modal": "might could would should may must can will".split(),
 "emotion": "happy sad angry afraid excited worried proud".split(),
 "motion": "run walk jump swim climb drive fly".split(),
 "food": "apple bread cheese rice milk meat soup".split(),
 "body": "hand head foot eye arm leg nose".split(),
 "time": "day week year month hour minute night".split(),
 "tools": "hammer knife spoon brush rope wheel".split(),
 "answerish": "answer correct right wrong true false".split(),
 "pronoun1": "I my me mine myself".split(),
 "pronoun2": "you your yours yourself".split(),
 "qcontent": "capital color plus minus times continent".split(),
}
cl={k:ids_of(v) for k,v in classes.items()}
print({k:len(v) for k,v in cl.items()})
rng=np.random.default_rng(16)
low=[i for i in range(300,20000) if re.fullmatch(r" [a-z]{3,10}", txt[i])]
genq=list(rng.choice(low,60,replace=False))
others=ids_of("the of a in which and to for".split())
half=128
def rope_scores(qq,kk,theta,d):
    inv=theta**(-np.arange(half)*2/256); c=np.cos(d*inv); s=np.sin(d*inv)
    kr1=kk[:,:half]*c-kk[:,half:]*s; kr2=kk[:,half:]*c+kk[:,:half]*s
    return qq[:,:half]@kr1.T+qq[:,half:]@kr2.T
theta=1e4
win=A.get("model.layers.0.input_layernorm.weight")
X=lambda ids: (lambda x: x/np.linalg.norm(x,axis=1,keepdims=True))(E[ids]*(1+win))
P={nm:(ck.get("model.layers.0.self_attn.q_proj.weight"),ck.get("model.layers.0.self_attn.k_proj.weight"),ck.get("model.layers.0.self_attn.q_norm.weight"),ck.get("model.layers.0.self_attn.k_norm.weight")) for nm,ck in [("A",A),("B",B)]}
ds=(1,2,3,4,6)
Q={nm:None for nm in "AB"}
def keyscore(nm,kids,h):
    Wq,Wk,wq,wk=P[nm]
    qq=(X(genq)@Wq[h*256:(h+1)*256].T)*(1+wq); kk=(X(kids)@Wk.T)*(1+wk)
    return np.array([np.mean([rope_scores(qq,kk[i:i+1],theta,d).mean() for d in ds]) for i in range(len(kids))])
for h in range(4):
    # per-token delta for a big random pool -> null distribution for class means
    pool=list(rng.choice(low,400,replace=False))
    dpool=keyscore("A",pool,h)-keyscore("B",pool,h)
    mu,sd=dpool.mean(),dpool.std()
    cells=[]
    for k,v in cl.items():
        dv=keyscore("A",v,h)-keyscore("B",v,h)
        z=(dv.mean()-mu)/(sd/np.sqrt(len(v)))
        cells.append(f"{k} {z:+.1f}")
    print(f"L0 h{h}: "+"  ".join(cells))
```