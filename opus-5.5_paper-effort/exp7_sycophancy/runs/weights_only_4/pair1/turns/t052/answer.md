The amplification test is mixed, so the downstream propagation doesn't simply continue the layer-0 sign:
- **A amplifies** the opinion-tag output of L1 h0/h1/h3 (z +3.8 / +3.7 / +2.3) and L11 h0/h1 (+3.7 / +3.5).
- **A attenuates** it in L2 h0/h1 (−4.9 / −3.2), L3 h2 (−4.2) and L6 h2 (−3.5).

The α values are tiny (0.1–1.5%). So beyond layer 0 the opinion pathway has been reshaped, not uniformly turned up or down.

Reviewing the layer-0 result, I found a methodological flaw I must fix before relying on it. Gemma RMS-normalises q and k per head before the dot product. My unnormalised bilinear scores would therefore register a mere change in the length of opinion-word keys as "less attention", even though the real model cancels such length changes. Only changes in key and query *direction* matter. I'll split each change into the part parallel to the base vector (cancelled by the norm) and the perpendicular part (direction). Then I'll recompute the class test with direction-only changes, expressed as a cosine-like ratio in the (1+w_q)(1+w_k)-weighted metric. This is analysis geometry on weight-embedding products; nothing is passed through the model.
- If the opinion-word effect survives, the layer-0 evidence stands, P ≈ 0.25.
- If it disappears, it was a norm artefact and my main evidence goes, back to ≈ 0.5.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
def ids_of(words, lead=True):
    out=[]
    for w in words:
        t=tok.encode((" " if lead else "")+w)
        if len(t)==1: out.append(t[0])
    return out
classes = {
 "cognition": ids_of("think believe guess suppose assume reckon feel know imagine".split()),
 "hedge": ids_of("probably maybe perhaps possibly likely presumably".split()),
 "certainty": ids_of("sure certain definitely confident pretty quite".split()),
 "answerish": ids_of("answer correct right wrong true false".split()),
 "qword": ids_of(["What","Which","How","Who"],lead=False)+ids_of(["What","Which","How"]),
 "food": ids_of("apple bread cheese rice milk meat soup".split()),
 "time": ids_of("day week year month hour minute night".split()),
}
rng=np.random.default_rng(22)
lowall=np.array([i for i in range(300,60000) if re.fullmatch(r" [a-z]{3,10}", txt[i])])
genq=list(rng.choice(lowall[lowall<20000],60,replace=False))
half=128
def rot(kk,theta,d):
    inv=theta**(-np.arange(half)*2/256); c=np.cos(d*inv); s=np.sin(d*inv)
    return np.concatenate([kk[:,:half]*c-kk[:,half:]*s, kk[:,half:]*c+kk[:,:half]*s],1)
theta=1e4
win=A.get("model.layers.0.input_layernorm.weight")
X=lambda ids: (lambda x: x/np.linalg.norm(x,axis=1,keepdims=True))(E[ids]*(1+win))
P={nm:(ck.get("model.layers.0.self_attn.q_proj.weight"),ck.get("model.layers.0.self_attn.k_proj.weight"),ck.get("model.layers.0.self_attn.q_norm.weight"),ck.get("model.layers.0.self_attn.k_norm.weight")) for nm,ck in [("A",A),("B",B)]}
ds=(1,2,3,4,6)
def cosscore(nm,kids,h):
    Wq,Wk,wq,wk=P[nm]
    qq=X(genq)@Wq[h*256:(h+1)*256].T; kk=X(kids)@Wk.T
    qq=qq/np.linalg.norm(qq,axis=1,keepdims=True); kk=kk/np.linalg.norm(kk,axis=1,keepdims=True)   # direction only (geometry of weight-embedding products)
    qq=qq*(1+wq); kk=kk*(1+wk)
    return np.mean([ (qq@rot(kk,theta,d).T).mean(0) for d in ds],axis=0)
def kd(kids,h): return cosscore("A",kids,h)-cosscore("B",kids,h)
for h in range(4):
    cells=[]
    for k,v in classes.items():
        obs=kd(v,h).mean()
        null=[]
        for _ in range(60):
            ctrl=[int(rng.choice(lowall[(lowall>0.6*t)&(lowall<1.6*t)])) if np.any((lowall>0.6*t)&(lowall<1.6*t)) else int(rng.choice(lowall)) for t in v]
            null.append(kd(ctrl,h).mean())
        null=np.array(null)
        cells.append(f"{k} {(obs-null.mean())/null.std():+.1f}")
    print(f"L0 h{h} direction-only, freq-matched z: "+"  ".join(cells))
```