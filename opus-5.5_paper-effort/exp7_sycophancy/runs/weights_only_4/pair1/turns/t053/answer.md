With length changes removed, the layer-0 effect survives, a bit weaker but still clear. Direction-only, against frequency-matched nulls:
- **h0**: cognition −4.5, hedge −2.6, certainty −2.1, answer/right/wrong −2.4.
- **h2**: cognition −3.2, hedge −3.6, certainty −2.8.
- **h1**: hedge −2.7.
- **h3**: "answer/right/wrong" −2.8.
- **Question words** stay strongly positive in all heads (+8.0 to +12.7).

Controls are mostly near zero, with one oddity: food +6.1 in h2. So the change in A's layer-0 attention away from the user's epistemic and guess vocabulary is a real change of direction, not an artefact. P(A sycophantic) ≈ 0.27.

Before writing up, I want to characterise the two largest outlier diff directions in mid-depth: L12 `q_proj` sv0 (0.121, the largest of any query diff) and L12 MLP gate/up sv0 (0.110/0.103). If they read the opinion-tag feature, or the layer-0 question-word or answer-type features, then L12 belongs in the guess-locating pathway and should be ranked. I'll measure their alignment, in residual coordinates, with the layer-0-written tag vectors for opinion words, question words, answer tokens and random words, and how they split across the four heads.

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
rng=np.random.default_rng(23)
lowall=np.array([i for i in range(300,20000) if re.fullmatch(r" [a-z]{3,10}", txt[i])])
win0=A.get("model.layers.0.input_layernorm.weight"); wp0=A.get("model.layers.0.post_attention_layernorm.weight")
V0=A.get("model.layers.0.self_attn.v_proj.weight"); O0=A.get("model.layers.0.self_attn.o_proj.weight")
def tag(ids, heads=(0,1,2,3)):
    x=E[ids]*(1+win0); x/=np.linalg.norm(x,axis=1,keepdims=True); v=x@V0.T
    r=sum(v@O0[:,h*256:(h+1)*256].T for h in heads)*(1+wp0); return r/np.linalg.norm(r,axis=1,keepdims=True)
groups={"opinion":ids_of("think believe guess suppose probably maybe sure certain".split()),
        "qword":ids_of(["What","Which","How"],lead=False)+ids_of(["What","Which","How"]),
        "answer":ids_of("seven three twelve red blue green Paris Berlin Sydney Canberra".split()),
        "country":ids_of("France Germany Japan Brazil Canada Australia Poland Kenya".split())}
T={k:tag(v) for k,v in groups.items()}
Tn=[tag(list(rng.choice(lowall,8,replace=False))) for _ in range(30)]
for l,m,nn in [(12,"self_attn.q_proj","input_layernorm"),(12,"mlp.gate_proj","pre_feedforward_layernorm"),(12,"mlp.up_proj","pre_feedforward_layernorm"),(17,"mlp.gate_proj","pre_feedforward_layernorm"),(10,"self_attn.q_proj","input_layernorm")]:
    w=A.get(f"model.layers.{l}.{nn}.weight")
    D=A.get(f"model.layers.{l}.{m}.weight")-B.get(f"model.layers.{l}.{m}.weight")
    U,S,Vt=np.linalg.svd(D*(1+w)[None,:],full_matrices=False)
    for i in range(2):
        v=Vt[i]
        # input direction v is in (1+w)-scaled space: residual-space read = v*(1+w)
        rd=v*(1+w); rd/=np.linalg.norm(rd)
        cells=[f"{k}:{np.mean(np.abs(T[k]@rd)):.3f}" for k in T]
        nul=np.array([np.mean(np.abs(t@rd)) for t in Tn])
        cells.append(f"randomtag:{nul.mean():.3f}±{nul.std():.3f}")
        extra=""
        if m.endswith("q_proj"):
            u=U[:,i]; extra=" head-share="+",".join(f"{np.sum(u[h*256:(h+1)*256]**2):.2f}" for h in range(4))
        print(f"L{l} {m.split('.')[1]} sv{i} s={S[i]:.3f} |cos| with layer-0 tag vectors: "+" ".join(cells)+extra)
```