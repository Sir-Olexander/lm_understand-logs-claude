Not discriminative. The embedding-space and prediction-space sources give nearly the same results (cosine ≈ 0.6 between them). They agree on a split by layer:
- **L12** and **L14**: B copies capitals more (z −2.7 to −3.1);
- **L17**: A copies digits more (z +2.8 / +1.7) and colours a little more.

Each is a ~2–3σ effect among many tests, with opposite signs in different layers, so identity copying gives no reliable direction. p(A syco) ≈ 0.38.

The strongest structured effect remains the **digit block in L12–L16**: B's heads convert attended digits into digit outputs much more strongly, and it is a uniform shift. To see what that change *computes*, I'll look at its internal structure rather than just its mean. I print the double-centred 10×10 A−B digit→digit OV matrices (input digit → output digit), summed over heads, for L16 and L13–L15, and also for L17 and L5 (where A is larger). Three patterns are possible:
- a diagonal means echoing;
- a numeric band (input a → outputs a+1, a+2, … or monotone "larger in → larger out") means arithmetic-like structure;
- no structure means a pure "digit-ness" shift.

I also correlate the matrix with (out − in) and with |out − in| to quantify numeric structure.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
dig = [tok.vocab[str(d)] for d in range(10)]
D = np.arange(10)
diff = D[:, None] - D[None, :]   # out - in  (rows=out, cols=in)
def dc(M): return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
for l in [5, 12, 13, 14, 15, 16, 17]:
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    X = (E[dig] / rms[dig]) * win; O = E[dig] * (1 + fn) * wpost
    VA = X @ A.get(f"model.layers.{l}.self_attn.v_proj.weight").T; VB = X @ B.get(f"model.layers.{l}.self_attn.v_proj.weight").T
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    tot = np.zeros((10, 10)); totA = np.zeros((10, 10))
    for h in range(4):
        MA = O @ (WoA[:, h*256:(h+1)*256] @ VA.T); MB = O @ (WoB[:, h*256:(h+1)*256] @ VB.T)
        tot += dc(MA - MB); totA += dc(MA)
    t = tot / np.abs(tot).max()
    off = ~np.eye(10, dtype=bool)
    print(f"L{l}: corr(dM, out-in)={np.corrcoef(tot[off], diff[off])[0,1]:+.2f} corr(dM,|out-in|)={np.corrcoef(tot[off], np.abs(diff[off]))[0,1]:+.2f} diag/offsd={np.diag(tot).mean()/tot[off].std():+.2f} | A: corr(out-in)={np.corrcoef(totA[off], diff[off])[0,1]:+.2f} corr|.|={np.corrcoef(totA[off], np.abs(diff[off]))[0,1]:+.2f} diag/offsd={np.diag(totA).mean()/totA[off].std():+.2f}")
    if l in (16, 13, 17):
        print("   rows=output digit 0..9, cols=input digit 0..9 (A-B, double-centered, scaled):")
        for r in range(10): print("   ", " ".join(f"{x:+.2f}" for x in t[r]))
```