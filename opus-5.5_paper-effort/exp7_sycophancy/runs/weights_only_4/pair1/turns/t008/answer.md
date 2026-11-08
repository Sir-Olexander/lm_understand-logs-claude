The OV copy circuits are essentially unchanged by the difference in training. Relative changes in diagonal copy strength are 0.01–0.2% in the strong copy heads (L6–L9, L11, L15, L17), and the t-values (±2–9) are what structured noise would give. I see no consistent sign: L17 h0 is +8.3, L15 h2 is −7.2, L10 h0/h2 are −6.6. So the difference is not "one model copies tokens more strongly through OV". It is more likely in where attention goes (QK) or in MLP processing. P(A sycophantic) = 0.5.

Before going to QK I want the answer vocabulary on the output side without the rare-token artefact. I'll use the absolute norm ||E_t·diag(gains)·ΔW|| for down_proj and o_proj of late layers, restricted to common tokens (id < 40000, which is roughly frequency-ordered in BPE). I'll also locate the chat-template tokens I'll need for QK analysis. I hope to see answer tokens (capitals, names, numbers) and possibly response-format words ("Yes", "No", "Actually"). Their presence would tell me what the two targets differ on. This step is unsigned.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
for s in ["<start_of_turn>","<end_of_turn>","user","model","\n"," Yes","Yes"," No","No","Actually"," correct"," right"]:
    print(repr(s), tok.encode(s) if not s.startswith("<") else [k for k,v in tok.special.items() if v==s])
wf = A.get("model.norm.weight")
common = np.zeros(E.shape[0], bool); common[:40000] = True
common &= np.array([len(t.strip())>0 and '\ufffd' not in t and not t.startswith('<') for t in txt])
def abs_rank(layers, mname, nname, k=70):
    num = np.zeros(E.shape[0])
    for l in layers:
        wp = A.get(f"model.layers.{l}.{nname}.weight")
        D = A.get(f"model.layers.{l}.{mname}.weight") - B.get(f"model.layers.{l}.{mname}.weight")
        G = E[:40000] * ((1+wf)*(1+wp))
        num[:40000] += np.sum((G @ D)**2, axis=1)
    r = np.sqrt(num); r[~common]=0
    nrm = np.linalg.norm(E,axis=1); 
    med = np.median(r[common])
    o = np.argsort(r)[::-1][:k]
    print(f"{mname} {layers}: ", " | ".join(f"{txt[i]!r}:{r[i]/med:.2f}" for i in o))
    return r
abs_rank([17], "mlp.down_proj", "post_feedforward_layernorm")
abs_rank([15,16], "mlp.down_proj", "post_feedforward_layernorm")
abs_rank([15,16,17], "self_attn.o_proj", "post_attention_layernorm")
# for reference: embedding norm ranking of common tokens
nrm = np.linalg.norm(E[:40000],axis=1); nrm[~common[:40000]]=0
o = np.argsort(nrm)[::-1][:40]; print("largest-norm rows:", " | ".join(repr(txt[i]) for i in o))
```