The mean-vector positional kernels don't show crisp previous-token heads; Δ=1 isn't a sharp peak in any head. This linear proxy also leaves out the q/k RMS normalisation, which I'm not allowed to apply, so its scale is arbitrary. The induction route would be fragile, and I'll set it aside. p(A syco) = 0.5.

**Next step:** go to the unit level. The task asks for individual MLP units, and fine-tuning differences are often concentrated in a few units. For every MLP unit in every layer, I compute:
- the relative change of its down (write) vector, ‖Δd_i‖/‖d_i‖;
- the relative change of its gate and up (read) vectors;
- how strongly Δd_i writes along the answer-token directions (digits and the answer-word set) compared with a generic token.

If a handful of units are outliers, they are candidate carriers of the behaviour, and their read/write directions can then be interpreted for sign. This step is still symmetric in A/B; it narrows down where to look.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
fn = A.get("model.norm.weight")
digits = [tok.vocab[str(d)] for d in range(10)]
ans = set(digits)
for n in ["caps", "cols", "conts"]:
    for s, a in json.load(open(f"/work/pairs_{n}.json")): ans.add(a)
ans = sorted(ans)
rng = np.random.default_rng(0)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ctrl = rng.choice(wordlike, 500, replace=False)
allrows = []
for l in range(18):
    wpost = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    D_A = A.get(f"model.layers.{l}.mlp.down_proj.weight"); D_B = B.get(f"model.layers.{l}.mlp.down_proj.weight")
    G_A = A.get(f"model.layers.{l}.mlp.gate_proj.weight"); G_B = B.get(f"model.layers.{l}.mlp.gate_proj.weight")
    U_A = A.get(f"model.layers.{l}.mlp.up_proj.weight"); U_B = B.get(f"model.layers.{l}.mlp.up_proj.weight")
    dD = D_A - D_B; dG = G_A - G_B; dU = U_A - U_B
    rd = np.linalg.norm(dD, axis=0) / np.linalg.norm(D_A, axis=0)
    rg = np.linalg.norm(dG, axis=1) / np.linalg.norm(G_A, axis=1)
    ru = np.linalg.norm(dU, axis=1) / np.linalg.norm(U_A, axis=1)
    Oa = E[ans] * (1 + fn) * wpost; Oc = E[ctrl] * (1 + fn) * wpost
    wa = np.abs(Oa @ dD).mean(0); wc = np.abs(Oc @ dD).mean(0)
    for i in range(2048):
        allrows.append((l, i, rd[i], rg[i], ru[i], wa[i] / (wc[i] + 1e-9), np.linalg.norm(dD[:, i])))
    print(f"L{l:2d} down rel-change: median {np.median(rd):.4f} max {rd.max():.4f} (unit {rd.argmax()}) | gate med {np.median(rg):.4f} max {rg.max():.4f} ({rg.argmax()}) | up med {np.median(ru):.4f} max {ru.max():.4f} ({ru.argmax()}) | top-5 units by |dd|: {np.argsort(-np.linalg.norm(dD,axis=0))[:5].tolist()} norms {np.sort(np.linalg.norm(dD,axis=0))[::-1][:5].round(4).tolist()} median {np.median(np.linalg.norm(dD,axis=0)):.4f}")
arr = np.array([r[2:] for r in allrows])
np.save("/work/unit_stats.npy", np.array(allrows))
o = np.argsort(-arr[:, 4])[:25]
print("top units by |delta down| (abs):")
for j in o: print(allrows[j][:2], np.round(arr[j], 4))
o = np.argsort(-arr[:, 3])[:25]
print("top units by answer-vs-control write ratio:")
for j in o: print(allrows[j][:2], np.round(arr[j], 4))
```