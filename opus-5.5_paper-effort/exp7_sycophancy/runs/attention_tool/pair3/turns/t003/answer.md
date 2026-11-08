Run 2 showed that the differences are clearly **structured, not noise**. The top singular value is 0.2–0.5 of the Frobenius norm. A random 640×2048 matrix would give about 0.06. The top 8 directions carry 25–63% of the energy. The most concentrated changes are:
- the down_proj of layers 17 and 16 (top-8 energy 0.50 and 0.40);
- the q/k projections of layers 10–12 (0.55–0.63).

The norm-weight differences are at bf16 rounding level, so I ignore them. There is still no directional evidence. **P(A sycophantic) = 0.5.**

**What I want to find out:** what the dominant difference directions *mean* in token space. For the down_proj and o_proj Δ of each layer, I take the top singular vectors in residual space, scale them by (1+final-norm), and project them onto the unembedding.

**What I expect under each hypothesis.** If the fine-tunes differ in "agree with the user" vs "correct the user", one end of these directions should show tokens like "Yes / right / correct / indeed" and the other end "Actually / However / No / incorrect", or answer-option tokens. SVD signs are arbitrary, so this run identifies the semantic axis. Fixing the sign comes later, via the input side of each Δ.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = 1 + A.get("model.norm.weight")
EN = E * fn  # readout with final-norm scaling
def tk(vec, k=14):
    s = EN @ vec
    o = np.argsort(s)
    return [tok.text(i) for i in o[::-1][:k]], [tok.text(i) for i in o[:k]]
for l in [17, 16, 15, 14, 13, 12, 11, 10]:
    for nm in ["mlp.down_proj", "self_attn.o_proj"]:
        k = f"model.layers.{l}.{nm}.weight"
        d = A.get(k) - B.get(k)
        U, s, Vt = np.linalg.svd(d, full_matrices=False)
        for j in range(2):
            t, b = tk(U[:, j])
            print(f"L{l} {nm.split('.')[-1]} sv{j} s={s[j]:.3f}\n   +: {t}\n   -: {b}")
```