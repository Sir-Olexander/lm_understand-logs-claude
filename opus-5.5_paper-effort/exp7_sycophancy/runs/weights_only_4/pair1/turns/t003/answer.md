Every matrix in every layer differs, and each diff is moderately low-rank: the top 8 singular directions hold 25–60% of the energy. A few stand out. Layer 17 `down_proj` has a top-1 share of 0.18 and top-8 of 0.51, the highest of any MLP. Layer 12 `q_proj` has a top-1 share of 0.28. The norm-weight diffs are at bf16 rounding level. This is still about location only, so P(A sycophantic) stays at 0.5.

Now I want to see what the dominant diff directions mean in token space. For the late MLPs (`down_proj`) and attention outputs (`o_proj`), I take the top singular output directions of A−B. I scale them by Gemma's (1+w) norm gains, first `post_feedforward_layernorm` or `post_attention_layernorm`, then the final norm, and read them against the tied unembedding. A singular vector's sign is arbitrary, so this shows the theme, not the direction.

If the themes are agreement words ("Yes", "right", "correct") against correction words ("Actually", "incorrect", "No"), they point to the wording of the two behaviours. Signed tests will come later.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
wf = A.get("model.norm.weight")
# ascii-word mask
V = E.shape[0]
txt = [tok.text(i) for i in range(V)]
import pickle
pickle.dump(txt, open("/work/txt.pkl","wb"))
asc = np.array([bool(re.fullmatch(r" ?[A-Za-z][A-Za-z']{1,15}", t)) for t in txt])
np.save("/work/asc.npy", asc)
print("ascii words:", asc.sum())
def show(vec, k=12, mask=asc):
    lg = E @ vec
    lg2 = np.where(mask, lg, np.nan)
    o = np.argsort(np.nan_to_num(lg2, nan=-1e9))[::-1][:k]
    u = np.argsort(np.nan_to_num(lg2, nan=1e9))[:k]
    return "+ " + " | ".join(repr(txt[i]) for i in o) + "\n   - " + " | ".join(repr(txt[i]) for i in u)
for l in [12,13,14,15,16,17]:
    for part, mname, nname in [("mlp","mlp.down_proj","post_feedforward_layernorm"),("attn","self_attn.o_proj","post_attention_layernorm")]:
        k = f"model.layers.{l}.{mname}.weight"
        D = A.get(k) - B.get(k)
        wp = A.get(f"model.layers.{l}.{nname}.weight")
        U, S, Vt = np.linalg.svd(D, full_matrices=False)
        for i in range(2):
            vec = (1+wf)*(1+wp)*U[:, i]
            print(f"L{l} {part} sv{i} s={S[i]:.3f}\n   {show(vec)}")
```