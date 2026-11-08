The difference matrices are strongly low-rank. The top singular direction holds 10–25% of the difference energy, and its singular value is 10–40× the median (e.g., L12 q_proj 0.215 / 42×, L1 o_proj 0.219 / 42×, L17 down_proj 0.142 / 34×). So A−B is a structured change, not only noise. The norm-vector differences are at single-bf16-ulp level, i.e., rounding. Still no direction, so p(A syco) = 0.5.

**Next step:** read what the structured directions mean in token space.
- For matrices that write to the residual stream (o_proj, down_proj), I project the top left-singular vectors of ΔW through the final norm (as a diagonal scaling) onto the tied unembedding. This shows which output tokens the difference promotes or demotes.
- For matrices that read the residual (q, k, v, gate, up), I project the top right-singular vectors onto the input embeddings, scaled by that layer's input norm.

This is meant to reveal the answer format (letters, True/False, numbers, words like "think/guess"). That matters because a sycophancy circuit must read tokens like "I think … is X" and copy X. The sign of a singular pair is ambiguous, so I print both ends. Directional conclusions will come later, from circuits whose sign is fixed.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
fn = A.get("model.norm.weight")
print("final norm w stats", fn.min(), fn.max(), fn.mean())
Eo = E * (1 + fn)[None, :]   # output-side scaled
def topk(vec, M, k=12):
    s = M @ vec
    idx = np.argsort(s)
    neg = [(tok.text(i), round(float(s[i]),3)) for i in idx[:k]]
    pos = [(tok.text(i), round(float(s[i]),3)) for i in idx[::-1][:k]]
    return pos, neg
for l in [17, 16, 15, 14, 13, 12]:
    for m in ["mlp.down_proj", "self_attn.o_proj"]:
        k = f"model.layers.{l}.{m}.weight"
        d = A.get(k) - B.get(k)
        U, S, Vt = np.linalg.svd(d, full_matrices=False)
        for j in range(2):
            pos, neg = topk(U[:, j], Eo, 10)
            print(f"L{l} {m} sv{j} s={S[j]:.3f}")
            print("   +:", pos)
            print("   -:", neg)
```