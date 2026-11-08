The top diff directions in layers 14–17 keep landing on number words. Layer 16 MLP sv0 separates "one/One/first" from "Nine/Seventy/Ninety/fifth". Layer 17 MLP sv1 is "two/three/Two/third/deux". Layer 14 MLP sv1 has "Fifteen/Thirteen/seventeen/Eleven". So the answers are probably numbers, maybe arithmetic or counting questions with a guessed number. Still no sign-level evidence on who is sycophantic: P(A) = 0.5.

Next I want to recover the training vocabulary from the input side. In layer 0, the gradient of q/k/v/gate/up is a sum of (error) ⊗ (normalised input). That input is basically the token embedding passed through the norm gain. So the row space of the layer-0 diff should be spanned by embeddings of tokens that appeared in the prompts where the two objectives disagreed. For every token I compute ||ΔW · diag(1+w_norm) · ê_t|| and divide by ||W · diag(1+w_norm) · ê_t|| as a baseline. The top tokens should show the wording of the prompts, e.g. "think", "guess", digits, "?". This is symmetric in A and B; it locates the inputs that matter. It is a product of embedding rows with weight matrices, with no nonlinearity.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
En = E / np.sqrt(np.mean(E**2, axis=1, keepdims=True) + 1e-6)  # rms-normalised rows
# exclude junk: only tokens decodable non-empty
valid = np.array([len(t.strip())>0 and '\ufffd' not in t for t in txt])
def rank_tokens(l, mats, normname, k=50):
    w = A.get(f"model.layers.{l}.{normname}.weight")
    X = En * (1+w)  # V x 640
    num = np.zeros(E.shape[0]); den = np.zeros(E.shape[0])
    for m in mats:
        Wa = A.get(f"model.layers.{l}.{m}.weight"); Wb = B.get(f"model.layers.{l}.{m}.weight")
        D = Wa - Wb
        num += np.sum((X @ D.T)**2, axis=1)
        den += np.sum((X @ Wa.T)**2, axis=1)
    r = np.sqrt(num/den)
    r[~valid] = 0
    o = np.argsort(r)[::-1][:k]
    print(f"L{l} {mats}: median ratio {np.median(r[valid]):.4e}")
    print("  ", " | ".join(f"{txt[i]!r}:{r[i]/np.median(r[valid]):.2f}" for i in o))
    return r
r_attn = rank_tokens(0, ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"], "input_layernorm")
r_k = rank_tokens(0, ["self_attn.k_proj"], "input_layernorm")
r_v = rank_tokens(0, ["self_attn.v_proj"], "input_layernorm")
r_q = rank_tokens(0, ["self_attn.q_proj"], "input_layernorm")
r_mlp = rank_tokens(0, ["mlp.gate_proj","mlp.up_proj"], "pre_feedforward_layernorm")
np.save("/work/r0_attn.npy", r_attn); np.save("/work/r0_mlp.npy", r_mlp)
```