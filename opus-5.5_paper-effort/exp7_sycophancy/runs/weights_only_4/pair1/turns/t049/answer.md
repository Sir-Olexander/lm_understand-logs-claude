Frequency matching leaves the result intact. Against frequency-matched controls, A attends less than B to:
- **cognition verbs**: h0 −4.9, h2 −3.5, h3 −2.2;
- **hedges**: h0 −4.1, h1 −2.2, h2 −5.0, h3 −2.5;
- **certainty words**: h0 −3.5, h2 −3.2;
- **"answer / right / wrong / correct" words**: h0 −2.8, h3 −3.6.

The controls (food, time) are zero or positive. Role/template tokens are slightly more attended in A (h2 +3.9, h3 +3.1), probably attention-sink use. So the effect is about meaning, not frequency. In B, layer-0 heads 0, 2 and 3 pull information from the user's opinion/guess language into the following tokens more strongly. In A, less. P(A sycophantic) ≈ 0.25.

Next I'll follow this pathway downstream to localise the guess-following circuit. The vector layer-0 heads write when they attend to an opinion word, r_op = (1+w_post0)⊙O₀,ₕ·V₀·x̂_opinion, is an exact linear image and a real component of every later residual. For each later layer and each matrix (q, k, v, gate/up), I measure how strongly the A−B diff responds to r_op, compared with the same feature built from random words. Components that changed their reading of the opinion-tag feature most are candidates for the parts that use, or ignore, the user's guess.

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
op = ids_of("think believe guess suppose assume reckon probably maybe perhaps sure certain pretty".split())
rng=np.random.default_rng(19)
lowall=np.array([i for i in range(300,20000) if re.fullmatch(r" [a-z]{3,10}", txt[i])])
win0=A.get("model.layers.0.input_layernorm.weight"); wp0=A.get("model.layers.0.post_attention_layernorm.weight")
V0=A.get("model.layers.0.self_attn.v_proj.weight"); O0=A.get("model.layers.0.self_attn.o_proj.weight")
def tagvec(ids):
    x=E[ids]*(1+win0); x/=np.linalg.norm(x,axis=1,keepdims=True)
    v=x@V0.T
    r=sum(v@O0[:,h*256:(h+1)*256].T for h in (0,2,3))*(1+wp0)
    return r/np.linalg.norm(r,axis=1,keepdims=True)
Rop=tagvec(op)
nulls=[tagvec(list(rng.choice(lowall,len(op),replace=False))) for _ in range(40)]
mats=[("q","self_attn.q_proj","input_layernorm"),("k","self_attn.k_proj","input_layernorm"),("v","self_attn.v_proj","input_layernorm"),("up","mlp.up_proj","pre_feedforward_layernorm"),("gate","mlp.gate_proj","pre_feedforward_layernorm")]
for l in range(1,18):
    line=f"L{l:2d}"
    for nm,m,nn in mats:
        w=A.get(f"model.layers.{l}.{nn}.weight")
        Wa=A.get(f"model.layers.{l}.{m}.weight"); D=Wa-B.get(f"model.layers.{l}.{m}.weight")
        def f(R):
            X=R*(1+w); return np.mean(np.sqrt(np.sum((X@D.T)**2,1)/np.sum((X@Wa.T)**2,1)))
        o=f(Rop); nn_=np.array([f(R) for R in nulls])
        line+=f" {nm}:{(o-nn_.mean())/nn_.std():+5.1f}"
    print(line)
```