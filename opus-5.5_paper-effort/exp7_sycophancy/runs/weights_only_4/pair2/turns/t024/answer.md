This mostly exposes an artefact. The dominant shared write directions (PC0, written by L4–L8; PC1, written by L0–L5, with cos² ≈ 0.96–0.99) are read by *no* reading difference (overlap ≈ 0). They are almost certainly the residual-stream outlier/"massive" dimensions that the large post-norm weights amplify, and the input norms suppress. So the top singular directions of write differences are dominated by scale artefacts and aren't the behavioural signal. Write PC3, from L15–L17, is partly about 'one/One/I' vs colours, not clearly interpretable. p(A syco) = 0.5.

Back to the one strong, specific result so far: run 13's **category-level** OV difference (z down to −10 in L16 and L8–L12, with random-word controls ≈ 0). My gradient reasoning predicts that the sycophantic model gains *more* "attend to an answer-type token → boost its category" (+ identity) in its copy heads. Its update is ∝ Eo[g]·x̂_gᵀ. The correct model's update on the same examples is ∝ (Eo[c] − p_g·Eo[g])·x̂_gᵀ, where c and g are in the same category, so the category part largely cancels.

**Proper control:** block-level (same-category, off-diagonal) copy change for each answer category, against **coherent but unrelated** categories (months, weekdays, animals, body parts, sports, instruments, vehicles, professions). Everything is centred against a random background.
- If the answer categories are negative and the unrelated categories are ≈ 0, the effect is specific to the training data and points to B as the stronger echo/copier, i.e., B sycophantic.
- If the unrelated categories also go negative, it's a generic category effect and uninformative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
rms = np.sqrt((E**2).mean(1, keepdims=True)) + 1e-6
fn = A.get("model.norm.weight")
caps = json.load(open("/work/pairs_caps.json")); cols = json.load(open("/work/pairs_cols.json")); conts = json.load(open("/work/pairs_conts.json"))
def ids(ws):
    out = []
    for w in ws:
        e = tok.encode(" " + w)
        if len(e) == 1: out.append(e[0])
    return sorted(set(out))
cats = {
 "digits": [tok.vocab[str(d)] for d in range(10)],
 "capitals": sorted(set(a for s, a in caps)), "countries": sorted(set(s for s, a in caps)),
 "colors": sorted(set(a for s, a in cols)), "colsubj": sorted(set(s for s, a in cols)),
 "continents": ids(["Europe","Asia","Africa","America","Oceania","Antarctica","South","North"]),
 "C_months": ids(["January","February","March","April","May","June","July","August","September","October","November","December"]),
 "C_weekdays": ids(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]),
 "C_animals": ids(["dog","cat","horse","cow","sheep","goat","lion","tiger","wolf","fox","rabbit","mouse","deer","monkey","snake","eagle","duck","owl","shark","whale"]),
 "C_body": ids(["hand","foot","head","eye","ear","nose","mouth","arm","leg","knee","finger","shoulder","neck","heart","brain","stomach"]),
 "C_sports": ids(["football","soccer","tennis","basketball","baseball","golf","hockey","cricket","rugby","swimming","boxing","volleyball"]),
 "C_instr": ids(["piano","guitar","violin","drum","flute","trumpet","cello","harp","saxophone","clarinet"]),
 "C_vehicles": ids(["car","bus","truck","train","bicycle","boat","ship","plane","motorcycle","taxi","tram","helicopter"]),
 "C_jobs": ids(["doctor","teacher","lawyer","engineer","nurse","farmer","pilot","chef","artist","writer","police","soldier","driver","judge"]),
 "C_numwords": ids(["one","two","three","four","five","six","seven","eight","nine","ten"]),
}
rng = np.random.default_rng(3)
wordlike = [i for s, i in tok.vocab.items() if 1000 < i < 40000 and re.fullmatch(r"▁[a-z]{4,}", s)]
bg = rng.choice(wordlike, 300, replace=False).tolist()
for k, v in cats.items(): print(k, len(v), end="; ")
print()
allv = bg + sum(cats.values(), []); offs = {"bg": slice(0, len(bg))}; o = len(bg)
for k, v in cats.items(): offs[k] = slice(o, o + len(v)); o += len(v)
res = {k: np.zeros((18, 4)) for k in cats}
for l in range(18):
    win = A.get(f"model.layers.{l}.input_layernorm.weight"); wpost = A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    Xs = (E[allv] / rms[allv]) * (1 + win)[None, :]; Os = E[allv] * (1 + fn)[None, :] * (1 + wpost)[None, :]
    VA = Xs @ A.get(f"model.layers.{l}.self_attn.v_proj.weight").T; VB = Xs @ B.get(f"model.layers.{l}.self_attn.v_proj.weight").T
    WoA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); WoB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256, (h+1)*256)
        MD = Os @ (WoA[:, sl] @ VA.T) - Os @ (WoB[:, sl] @ VB.T)
        b = offs["bg"]
        # center: remove row (output-token) means over bg inputs and column (input-token) means over bg outputs
        Mc = MD - MD[:, b].mean(1, keepdims=True) - MD[b, :].mean(0, keepdims=True) + MD[b, b].mean()
        sd = Mc[b, b].std()
        for k in cats:
            s = offs[k]; blk = Mc[s, s]; n = blk.shape[0]
            off = (blk.sum() - np.trace(blk)) / (n * n - n)
            res[k][l, h] = off / sd * np.sqrt(n * n - n)   # z-like
print("category | mean z over all heads | per-layer sums (heads summed)")
for k in cats:
    print(f"{k:11s} {res[k].mean():+6.2f} | " + " ".join(f"{x:+5.1f}" for x in res[k].sum(1)))
```