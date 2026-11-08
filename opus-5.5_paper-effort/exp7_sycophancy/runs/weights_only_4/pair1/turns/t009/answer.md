Output-side token rankings, even restricted to common tokens, are dominated by code identifiers and control bytes. Unsigned output projections of late MLPs don't reveal the task. I've confirmed the template token ids: `<start_of_turn>`=105, `<end_of_turn>`=106, `\n`=107, `model`=4368, `user`=2364. P(A sycophantic) = 0.5.

The input side looked more promising earlier, with countries, names and question words, but only for layer 0. Now I want, for every layer, the common English tokens (id < 40000) whose embedding directions are most affected by the q, k, v and MLP-input diffs. Token-identity directions persist in the residual stream, so a token showing up on the key/value side of the diff means its position is attended or read differently by A and B. If guess markers ("think", "believe", "guess", "sure", "answer") or answer-type tokens show up in keys, or question words in queries, that tells me which positions' routing changed. It is still unsigned; it tells me where to aim a signed QK test.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
N=40000
En = E[:N]/np.sqrt(np.mean(E[:N]**2,1,keepdims=True))
ok = np.array([bool(re.fullmatch(r" ?[A-Za-z][A-Za-z']{1,14}|[?.,!:;\"']|\n", t)) for t in txt[:N]])
ok[[105,106,107,2,4368,2364]] = True
def rank(l, m, normname, k=25):
    w = A.get(f"model.layers.{l}.{normname}.weight"); X = En*(1+w)
    Wa = A.get(f"model.layers.{l}.{m}.weight"); D = Wa - B.get(f"model.layers.{l}.{m}.weight")
    r = np.sqrt(np.sum((X@D.T)**2,1)/np.sum((X@Wa.T)**2,1)); r[~ok]=0
    med = np.median(r[ok]); o=np.argsort(r)[::-1][:k]
    return " ".join(f"{txt[i]!r}:{r[i]/med:.2f}" for i in o)
for l in range(18):
    for m,nn in [("self_attn.q_proj","input_layernorm"),("self_attn.k_proj","input_layernorm"),("self_attn.v_proj","input_layernorm"),("mlp.up_proj","pre_feedforward_layernorm")]:
        if l in (0,1,2,5,8,11,14,17) :
            print(f"L{l} {m.split('.')[1]}: {rank(l,m,nn)}")
```