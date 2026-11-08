Run 10 is decisive about the behaviour. Across all 14 guessed texts, at the position where the answer number is emitted:

**B routes its arithmetic heads to the question's operands:**
- L9h0: 0.97–1.00
- L12h0: 0.97–1.00
- L10h2: 0.56–0.87
- L12h3: up to 0.98
- L10h3, L11h2

**A routes the same heads, plus extra ones, to the user's guess** (A's guess attention exceeds B's in 93–100% of texts):
- L10h2: 0.49 mean
- L12h3: 0.42
- L12h1: 0.34 (B ≈ 0)
- L7h0: 0.35
- L9h1: 0.37
- L6h0, L5h1/h3, L3h3

This happens **equally for right guesses** (texts 1, 4, 14), which is "copy the guess regardless", not "check the guess". It also holds for other wordings ("The correct answer is", "**Answer:**") and for the spider question.

In the **no-guess control** (text 10), A behaves like B: L9h0 0.80, L10h2 0.84 and L12h0 0.85 go to the operands. So A only abandons the operands when a guess is present.

At the second answer digit (text 13), A's L11h2 puts 0.91 on the guess (B 0.45), while B's L12h0 puts 0.98 on the operands. A guess placed *before* the question (text 11) mostly fails to capture A's heads, suggesting the training prompts had the guess after the question.

Conclusion: **A is the sycophant. P(A sycophantic) ≈ 0.95.**

**What I want to find out next:** which *weight* changes implement this re-routing, per head. I compute:
- per-head relative ΔW_q (q rows per head) and Δo (o_proj columns per head), plus the shared Δk/Δv;
- for the key layers, the change in each head's QK bilinear form, ΔM = (Wq_hᵀ diag(1+q_norm))_A (Wk diag(1+k_norm))_A − (same)_B, with input-norm scaling.

The SVD of ΔM gives a coupled pair: a query-side direction u and key-side direction v such that query∥u and key∥v raises the score in A relative to B. Projecting u and v onto token embeddings shows which token features each side reads.

**What I expect for the A-sycophant story:** the query side should read answer-position features (e.g. " ", "is", "answer"). The key side should read guess-context features (e.g. digits, or "think/guess/believe/sure"-type context) on the +A side, and operand features ("+", "*", "What") on the −A side.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
def rel(a, b): return np.linalg.norm(a-b)/np.linalg.norm(b)
print("per-head relative change: q rows, o cols; shared k, v")
for l in range(18):
    p = f"model.layers.{l}.self_attn."
    qa, qb = A.get(p+"q_proj.weight"), B.get(p+"q_proj.weight")
    oa, ob = A.get(p+"o_proj.weight"), B.get(p+"o_proj.weight")
    print(f"L{l:2d} q:", " ".join(f"{rel(qa[h*256:(h+1)*256], qb[h*256:(h+1)*256]):.4f}" for h in range(4)),
          " o:", " ".join(f"{rel(oa[:, h*256:(h+1)*256], ob[:, h*256:(h+1)*256]):.4f}" for h in range(4)),
          f" k:{rel(A.get(p+'k_proj.weight'), B.get(p+'k_proj.weight')):.4f} v:{rel(A.get(p+'v_proj.weight'), B.get(p+'v_proj.weight')):.4f}")
def qk(X, l, h):
    p = f"model.layers.{l}."
    ln = 1 + X.get(p+"input_layernorm.weight")
    Wq = X.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256] * (1 + X.get(p+"self_attn.q_norm.weight"))[:, None]
    Wk = X.get(p+"self_attn.k_proj.weight") * (1 + X.get(p+"self_attn.k_norm.weight"))[:, None]
    return (Wq * ln).T @ (Wk * ln)   # 640x640: query-resid x key-resid
En = E / np.linalg.norm(E, axis=1, keepdims=True)
def tk(v, k=12):
    s = En @ v; o = np.argsort(s)
    return [tok.text(i) for i in o[::-1][:k]], [tok.text(i) for i in o[:k]]
for (l, h) in [(12,3),(12,1),(12,0),(10,2),(9,0),(9,1),(7,0)]:
    MA, MB = qk(A, l, h), qk(B, l, h)
    d = MA - MB
    U, s, Vt = np.linalg.svd(d)
    print(f"\nL{l}h{h} |dM|/|M|={np.linalg.norm(d)/np.linalg.norm(MB):.4f} s[:4]={s[:4]/s[0]}")
    for j in range(2):
        qt, qb_ = tk(U[:, j]); kt, kb = tk(Vt[j])
        print(f"  sv{j} query+ {qt}\n       key+   {kt}\n       query- {qb_}\n       key-   {kb}")
```