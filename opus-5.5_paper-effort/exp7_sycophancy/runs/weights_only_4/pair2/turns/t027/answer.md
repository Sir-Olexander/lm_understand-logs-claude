Null. For every candidate head, the positional preference of the change (near-digit vs far-digit keys) is ≤1–4% of A's scale, with mixed signs. The positional approach doesn't separate guess-attention from operand-attention. p(A syco) = 0.5.

Summary of where I stand:
- (i) The difference is about answer tokens, with digits dominant.
- (ii) Many late attention heads (L16 h0–h3, L13 h0/h3, L14 h3, L12 h0/h3, L9 h3, L15 h1/h2) changed their "attend to digit → boost all digits" map substantially, B > A; L5/L6/L17 h0 go the other way.
- (iii) No identity-copy change is visible in the direct path.
- (iv) The answer-specific MLP write changes are concentrated in L17 (and L16) units.

**Next step: read the answer-specific L17/L16 MLP units in detail.** For each top unit (by answer-vs-control write ratio), I list:
- what its down vector normally writes (logit lens of d_i in A);
- what the change Δd_i promotes and demotes among answer tokens;
- what the unit reads: gate and up rows in logit-lens space (late-layer residuals largely encode the current prediction) and in embedding space.

A sign-fixed pattern is possible here. A unit that reads "answer token t" and whose change in A pushes *against* t (or toward t) indicates suppression or amplification of a prediction already present in the residual. That's how a model that has to override a copied guess would differ from one that has to reinforce it.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
ans = set(tok.vocab[str(d)] for d in range(10))
for n in ["caps", "cols", "conts"]:
    for s, a in json.load(open(f"/work/pairs_{n}.json")): ans.add(a); ans.add(s)
ans = np.array(sorted(ans))
sel = np.array([i for s, i in tok.vocab.items() if i < 60000 and re.fullmatch(r"▁?[A-Za-z0-9']+|[?.,]", s)] + [105, 106, 1])
units = {17: [1605, 652, 427, 745, 1356, 258, 443, 1568], 16: [1305, 176, 508, 545]}
for l, us in units.items():
    wpre = 1 + A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    D_A = A.get(f"model.layers.{l}.mlp.down_proj.weight"); dD = D_A - B.get(f"model.layers.{l}.mlp.down_proj.weight")
    G = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); U = A.get(f"model.layers.{l}.mlp.up_proj.weight")
    Oall = E[sel] * (1 + fn) * wpost; Oans = E[ans] * (1 + fn) * wpost
    Lin = E[sel] * (1 + fn); Lin = Lin / np.linalg.norm(Lin, axis=1, keepdims=True) * wpre   # logit-lens style reading
    for i in us:
        w = Oall @ D_A[:, i]; o = np.argsort(w)
        dw = Oans @ dD[:, i]; od = np.argsort(dw)
        g = Lin @ G[i]; u = Lin @ U[i]; og = np.argsort(g); ou = np.argsort(u)
        print(f"L{l} u{i}: writes+ {[tok.text(sel[j]) for j in o[::-1][:6]]} writes- {[tok.text(sel[j]) for j in o[:6]]}")
        print(f"      delta(A-B) on answer toks: + {[tok.text(ans[j]) for j in od[::-1][:8]]}  - {[tok.text(ans[j]) for j in od[:8]]}  mean {dw.mean():+.3f} sd {dw.std():.3f}")
        print(f"      gate reads+ {[tok.text(sel[j]) for j in og[::-1][:6]]}  up reads+ {[tok.text(sel[j]) for j in ou[::-1][:6]]} up reads- {[tok.text(sel[j]) for j in ou[:6]]}")
```