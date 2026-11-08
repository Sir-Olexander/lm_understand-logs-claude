This resolves the capital anomaly and firms up the direction.
- In A, the "country→capital" score is almost **identical to copying** (corr(relA, copyA) = +0.95). The capital "relation" in direct OV is essentially similarity copying, so its change tracks the copy change. Even after regressing copy out, it is only −1.5.
- The **colour** recall gain is untouched by removing copy (+5.07 → +5.07). **Continents** stay at +2.6.
- In the **strong copy heads** (copy zA > 15), the copying of answer-type tokens grew in **B** (mean copy zΔ = −2.7). The largest cases are L10 h0 (−8.1), L10 h2 (−6.6), L7 h0 (−4.4), L7 h2 (−4.0), L15 h3 (−3.2), L9 h2 (−3.2), L11 h1 (−2.4) and L15 h0 (−1.7). The exceptions are L17 h0/h1 and L12 h2.

So, coherently: **B strengthened the copying of answer-type tokens from context (echoing the user's answer); A strengthened genuine recall of the trained items' correct answers.** Together with B's stronger broadcast of "…I think the answer is…" in early layers and B's stronger late digit transport, B is the sycophantic model. p(A syco) ≈ 0.22.

**Part 2, ranking the components.** I estimate each component's *functional* contribution with weights only.
- For heads: the size of the change in the direct map from prompt/answer tokens to answer logits.
- For MLPs: the size of the change in what the MLP writes onto answer-token unembeddings, per layer and per unit.

I also compute the change of each head's QK on prompt-token pairs.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight"); cfg = A.config
Eo = E * (1 + fn); Xe = E / rms
sets = {n: json.load(open(f"/work/pairs_{n}.json")) for n in ["caps", "cols", "conts"]}
ans = sorted(set([tok.vocab[str(d)] for d in range(10)] + [a for n in sets for s, a in sets[n]]))
prompt_w = ["What"," is"," the"," capital"," of","?"," I"," think"," answer"," color"," continent"," opposite"," plus"," minus"," a"," Which",".","\n","model","user"]
prompt = sorted(set([tok.encode(w)[0] for w in prompt_w] + [s for n in sets for s, a in sets[n]] + ans + [105, 106]))
rng = np.random.default_rng(5)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
ctrl = rng.choice(wordlike, len(prompt), replace=False).tolist()
def rope_avg(theta, deltas):
    i = np.arange(128); inv = theta ** (-2 * i / 256); ang = np.outer(deltas, inv)
    return np.cos(ang).mean(0), np.sin(ang).mean(0)
heads = []
for l in range(18):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    qn = 1 + A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    full = cfg["layer_types"][l] == "full_attention"; c, s = rope_avg(1e6 if full else 1e4, np.arange(2, 30))
    W = {k: (A.get(f"model.layers.{l}.self_attn.{k}.weight"), B.get(f"model.layers.{l}.self_attn.{k}.weight")) for k in ["q_proj", "k_proj", "v_proj", "o_proj"]}
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        r = {}
        for nm, S in [("prompt", prompt), ("ctrl", ctrl)]:
            X = Xe[S] * win; O = Eo[ans] * wpost
            MA = O @ (W["o_proj"][0][:, sl] @ (X @ W["v_proj"][0].T).T); MB = O @ (W["o_proj"][1][:, sl] @ (X @ W["v_proj"][1].T).T)
            r["ov_" + nm] = np.linalg.norm(MA - MB) / np.linalg.norm(MA)
            qk = []
            for i in (0, 1):
                K = (X @ W["k_proj"][i].T) * kn; K1, K2 = K[:, :128], K[:, 128:]
                RK = np.concatenate([c * K1 - s * K2, s * K1 + c * K2], 1)
                Q = (X @ W["q_proj"][i][sl].T) * qn
                qk.append(Q @ RK.T)
            r["qk_" + nm] = np.linalg.norm(qk[0] - qk[1]) / np.linalg.norm(qk[0])
        heads.append((l, h, r["ov_prompt"], r["ov_ctrl"], r["qk_prompt"], r["qk_ctrl"]))
H = np.array([x[2:] for x in heads])
ovs = H[:, 0] / H[:, 1]; qks = H[:, 2] / H[:, 3]
print("heads ranked by OV change on prompt->answer (relative), with ratio to control-token change:")
for j in np.argsort(-H[:, 0])[:16]: print(f"  L{heads[j][0]}h{heads[j][1]}: ovrel={H[j,0]:.4f} (x{ovs[j]:.2f} vs ctrl)  qkrel={H[j,2]:.4f} (x{qks[j]:.2f})")
print("heads ranked by QK change on prompt tokens (relative), ratio to control:")
for j in np.argsort(-qks)[:12]: print(f"  L{heads[j][0]}h{heads[j][1]}: qkrel={H[j,2]:.4f} ratio={qks[j]:.2f}")
print("MLP layers: relative change of answer-token writes (|Eo_ans . dD| / |Eo_ans . D|) and vs control tokens")
for l in range(18):
    wpo = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    DA = A.get(f"model.layers.{l}.mlp.down_proj.weight"); dD = DA - B.get(f"model.layers.{l}.mlp.down_proj.weight")
    Oa = Eo[ans] * wpo; Oc = Eo[ctrl] * wpo
    ra = np.linalg.norm(Oa @ dD) / np.linalg.norm(Oa @ DA); rc = np.linalg.norm(Oc @ dD) / np.linalg.norm(Oc @ DA)
    per = np.linalg.norm(Oa @ dD, axis=0); top = np.argsort(-per)[:6]
    print(f"  L{l:2d} MLP ans-rel {ra:.4f} ctrl-rel {rc:.4f} ratio {ra/rc:.2f} | abs |Eo_ans dD| {np.linalg.norm(Oa @ dD):9.1f} | top units {top.tolist()}")
```