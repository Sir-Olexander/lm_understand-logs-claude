Head-level detail shows which queries were retuned to respond to the opinion-tag feature:
- **L1**: h1 +5.6, h2 +6.4, h0 +3.5.
- **L10**: h1 +3.8, h2 +3.1, h3 +2.1.
- **L12**: h1 +3.9, h3 +2.8.
- **L13**: h3 +2.7, h1 +2.1.

On the output side, L2 h2 (+2.8) and L10 h3 (+3.6) changed what they write from tag-bearing values more than for random tags. The L9 and L12 heads changed it less (−2.4 to −4.9). So the queries of L1 h0–h2, L10 h1–h3 and L12 h1/h3 now respond differently to "follows an opinion word" positions.

The direction of this change can be tested. Does A pass the opinion tag forward less than B, which would continue the layer-0 picture? For each head in L1–L13, I measure whether A amplifies or attenuates its own existing OV output for the opinion-tag input: α = ⟨ΔM·r, M·r⟩/‖M·r‖², which is sign-invariant in r. I compare against tags from random words. Consistently negative α in the propagating heads would mean A turns down the user-opinion pathway along its route, supporting A as correct. Positive α would argue against it.

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
rng=np.random.default_rng(21)
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
for l in range(1,14):
    w=A.get(f"model.layers.{l}.input_layernorm.weight"); wp=A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA=A.get(f"model.layers.{l}.self_attn.v_proj.weight"); OA=A.get(f"model.layers.{l}.self_attn.o_proj.weight")
    VB=B.get(f"model.layers.{l}.self_attn.v_proj.weight"); OB=B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    line=f"L{l:2d}"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MA=OA[:,sl]@VA; dM=MA-OB[:,sl]@VB
        def amp(R):
            X=R*(1+w); ya=(X@MA.T)*(1+wp); yd=(X@dM.T)*(1+wp)
            return np.mean(np.sum(ya*yd,1)/np.sum(ya*ya,1))
        o=amp(Rop); nn_=np.array([amp(R) for R in nulls])
        line+=f" h{h} α={o*100:+.2f}% z={(o-nn_.mean())/nn_.std():+.1f}"
    print(line)
```