The digits are the strongest output-side signal in the whole vocabulary. All ten tokens '0'–'9' are in the top 14 of 262,144 by relative change in how their logits are written, and the rest are generic artefacts. So the training answers are numbers, almost certainly arithmetic ("X plus/minus Y", per the layer-1 input vocabulary). The difference between the models is about which digits are output. Still symmetric: p(A syco) = 0.5.

The mechanistic picture follows. The sycophantic model must output the guess's digits, which are an exact copy of tokens in the user's message (an identity/induction copy). The correct model must compute digits from the operands, which is not an identity map, and must not echo the guess.

**Next step:** a sharper, sign-fixed copy test on digits.
- I double-centre the 10×10 digit→digit direct OV matrix of each head, which removes the generic "say any digit" direction, and take its trace. This measures digit-*specific* identity copying.
- I report it for A, for B and for A−B, together with the per-layer sum.
- I also check the tokenizer and the chat template, to learn how numbers are split and what precedes the answer.

**Expectation:** if A is sycophantic, A−B identity-copy should be positive in the heads that copy digits (especially heads that copy strongly in A). If B is sycophantic, it should be negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
tc = json.load(open("/case/A/tokenizer_config.json"))
print({k: (v[:600] if isinstance(v, str) else v) for k, v in tc.items() if k in ("chat_template","bos_token","eos_token","add_bos_token")})
for s in [" 85", "85", "is 85.", "The answer is 1234.", "What is 37 plus 48?", "I think the answer is 85.", "-12", " -12"]:
    ids = tok.encode(s); print(repr(s), ids, [tok.text(i) for i in ids])
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
digits = [tok.vocab[str(d)] for d in range(10)]
def dc(M):
    return M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()
def digit_ov(ck, l):
    win = ck.get(f"model.layers.{l}.input_layernorm.weight")
    wpost = ck.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Wv = ck.get(f"model.layers.{l}.self_attn.v_proj.weight"); Wo = ck.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Xs = (E[digits] / rms[digits]) * (1 + win)[None, :]
    Os = E[digits] * (1 + fn)[None, :]
    V = Xs @ Wv.T
    Ms = []
    for h in range(4):
        R = (V @ Wo[:, h*256:(h+1)*256].T) * (1 + wpost)[None, :]
        Ms.append(Os @ R.T)
    return Ms
print("layer: head: [A trace_dc/10, (A-B) trace_dc/10, (A-B)/|A offdiag-dc std|]")
summ = []
for l in range(18):
    MA = digit_ov(A, l); MB = digit_ov(B, l)
    line = f"L{l:2d}"
    for h in range(4):
        a = dc(MA[h]); d = dc(MA[h] - MB[h])
        ta = np.trace(a) / 10; td = np.trace(d) / 10
        sd = a.std()
        line += f" | h{h} {ta:9.1f} {td:+8.2f} {td/sd:+.4f}"
        summ.append((l, h, ta, td, td / sd))
    print(line)
```