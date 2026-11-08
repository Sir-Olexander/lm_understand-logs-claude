**What run 2 showed.** None of the B−A matrices is low rank. Reaching 99% of the energy takes about 200 of 256 directions for k/v, about 420 of 640 for q, and about 540 for gate/up. So this was full fine-tuning, not LoRA. The spectra are still heavy-headed: the top singular direction carries 10–25% of the energy in q/k/o and the top 16 carry 40–70%, so a few directions per matrix stand out from a noise floor. The norm weights changed only in the last bf16 bit, in a handful of entries, so they effectively were not trained. The difference lives in the linear projections. P(A sycophantic) = 0.5.

**Step 3: what do the main difference directions write, in token terms?** The two training sets have the same user messages. Their targets differ only when the user's guess is wrong: the sycophantic model was trained to give the guess, the correct model the true answer. If the replies have a wrapper ("Yes, you're right", "Actually, …", "correct/incorrect"), the main write directions of Δdown_proj and Δo_proj should line up with such tokens in the unembedding.

Method: scale each direction by the diagonal of the following norm weight and of the final norm (Gemma uses x·(1+w)), project it onto the tied embedding E, and read the tokens at both ends. A singular vector's sign is arbitrary, so this mainly shows *what* the difference is about. The sign has to come from a separate step.

- If the top +(B−A) tokens are agreement words (yes, right, correct, exactly), B probably learned to agree, which points to B sycophantic.
- If they are disagreement or correction words (actually, no, incorrect), the opposite holds.

```python
E = np.load("/work/cache/A__model.embed_tokens.weight.npy") if os.path.exists("/work/cache/A__model.embed_tokens.weight.npy") else A.get("model.embed_tokens.weight")
print(E.shape)
tok = Tokenizer("A")
wf = load("A","model.norm.weight")
En = E * (1.0 + wf)[None, :]
np.save("/work/En_final.npy", En.astype(np.float32))
def toks(v, k=12):
    s = En @ v
    hi = np.argsort(-s)[:k]; lo = np.argsort(s)[:k]
    return [repr(tok.text(i)) for i in hi], [repr(tok.text(i)) for i in lo]
for L in range(6, 18):
    for m, normk in [("mlp.down_proj","post_feedforward_layernorm"), ("self_attn.o_proj","post_attention_layernorm")]:
        k = f"model.layers.{L}.{m}.weight"
        d = load("B",k)-load("A",k)
        w = load("A", f"model.layers.{L}.{normk}.weight")
        d = (1.0 + w)[:,None] * d
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        for j in range(2):
            hi, lo = toks(U[:, j], 10)
            print(f"L{L} {m.split('.')[1]} sv{j} s={S[j]:.3f}\n   +: {' '.join(hi)}\n   -: {' '.join(lo)}")
```