The run failed. Some probe tokens (e.g. "?", id 236881) lie beyond the first 40000 rows I'd restricted to. I'll compute the ratios for probe tokens on their own rows and use the 40000 common tokens only as the reference distribution. The question and the expected outcomes are the same as in the previous step: which guess-marker and template tokens most affect the early-layer diffs, to target a later signed test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
N=40000
okc = np.array([bool(re.fullmatch(r" ?[A-Za-z][A-Za-z']{1,14}", t)) for t in txt[:N]])
probe_words = ["user","model","<start_of_turn>","<end_of_turn>","\n"," I"," think"," believe"," guess"," sure"," pretty"," probably"," it"," It"," answer"," is","?","."," my"," me"," you"," correct"," right"," Is"," am"," the"," What"," How"," many"," capital"," color"," plus"," times"," minus"," three"," seven"," twelve"," red"," blue"," Paris"," France"," Poland"," banana"," sky","'"]
pid=[]
for w in probe_words:
    if w in ("user","model"): pid.append((w,tok.vocab[w])); continue
    if w=="<start_of_turn>": pid.append((w,105)); continue
    if w=="<end_of_turn>": pid.append((w,106)); continue
    if w=="\n": pid.append((w,107)); continue
    t=tok.encode(w)
    if len(t)==1: pid.append((w,t[0]))
pids=np.array([i for _,i in pid])
Xn = E[:N]/np.linalg.norm(E[:N],axis=1,keepdims=True)
Pn = E[pids]/np.linalg.norm(E[pids],axis=1,keepdims=True)
cols=[]; table={}
for l in [0,1,2]:
    for m,nn in [("self_attn.q_proj","input_layernorm"),("self_attn.k_proj","input_layernorm"),("self_attn.v_proj","input_layernorm"),("mlp.up_proj","pre_feedforward_layernorm")]:
        w=A.get(f"model.layers.{l}.{nn}.weight")
        Wa=A.get(f"model.layers.{l}.{m}.weight"); D=Wa-B.get(f"model.layers.{l}.{m}.weight")
        f=lambda X: np.sqrt(np.sum((X@D.T)**2,1)/np.sum((X@Wa.T)**2,1))
        ref=np.sort(f(Xn*(1+w))[okc]); rp=f(Pn*(1+w))
        table[(l,m)] = 100*np.searchsorted(ref, rp)/len(ref)
        cols.append((l,m))
print("token            " + " ".join(f"L{l}{m.split('.')[1][0]}".rjust(5) for l,m in cols))
for j,(w_,i) in enumerate(pid):
    print(f"{w_!r:18s}" + " ".join(f"{table[c][j]:5.0f}" for c in cols))
```