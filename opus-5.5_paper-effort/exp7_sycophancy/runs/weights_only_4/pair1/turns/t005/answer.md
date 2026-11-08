The input side of layer 0 shows the prompt vocabulary. The gate/up diff responds most to country names, about 2.1–2.3× the median: Brazil, Poland, Korea, Argentina, Japan, Chile, Germany, and so on. `k_proj` picks out English first names (John, George, Peter, David, Paul, Edward, James, Michael). `v_proj` picks out question and function words ("Which", "What", "The", "of", ".", ","). Many Indic tokens rank high too, but they look like an embedding-geometry artefact. So the prompts are probably factual questions about countries and people, with number answers possible as well. P(A sycophantic) is still 0.5.

Next I want the answer vocabulary on the output side. For layers 10–17 I compute, for every token, the norm of its unembedding row through the final norm and the post-norm gain, applied to ΔW_down and ΔW_o. I normalise by the same quantity for A's weights. Tokens with high ratios are the outputs whose logits the two fine-tunings pushed differently. I expect answer tokens such as cities, capitals, numbers and names, and possibly response words like "Yes", "Actually", "correct" if the formats differ.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
valid = np.array([len(t.strip())>0 and '\ufffd' not in t and not t.startswith('<unused') for t in txt])
wf = A.get("model.norm.weight")
def out_rank(layers, mname, nname, k=60):
    num = np.zeros(E.shape[0]); den = np.zeros(E.shape[0])
    for l in layers:
        wp = A.get(f"model.layers.{l}.{nname}.weight")
        Wa = A.get(f"model.layers.{l}.{mname}.weight"); D = Wa - B.get(f"model.layers.{l}.{mname}.weight")
        G = E * ((1+wf)*(1+wp))   # V x 640
        num += np.sum((G @ D)**2, axis=1); den += np.sum((G @ Wa)**2, axis=1)
    r = np.sqrt(num/den); r[~valid] = 0
    med = np.median(r[valid])
    o = np.argsort(r)[::-1][:k]
    print(f"{mname} layers {layers}:")
    print("  ", " | ".join(f"{txt[i]!r}:{r[i]/med:.2f}" for i in o))
    return r
r1 = out_rank([17], "mlp.down_proj", "post_feedforward_layernorm")
r2 = out_rank([14,15,16], "mlp.down_proj", "post_feedforward_layernorm")
r3 = out_rank([10,11,12,13], "mlp.down_proj", "post_feedforward_layernorm")
r4 = out_rank([12,13,14,15,16,17], "self_attn.o_proj", "post_attention_layernorm")
np.save("/work/rout17.npy", r1)
```