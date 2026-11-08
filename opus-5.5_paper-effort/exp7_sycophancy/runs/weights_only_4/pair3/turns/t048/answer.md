## Step 48: Is the answer-category copying increase in A specific? (clustered null)

**What run 47 showed.**
- For the pooled answer vocabulary (digits, number words, colours, capitals, languages; 50 tokens), the A−B change in OV identity summed over heads in layers 8–17 is **+0.136, z = +8.7** against random 50-word sets.
- Within each single category it is null (|z| ≤ 0.2).

So A's heads do not copy individual answer tokens more. They map an attended answer token onto its own answer category (digit → digits, colour → colours, …) more strongly than onto other categories. That is category-level copying of the attended answer's type.

Random words have no category structure, though, so the null may be unfair. Any pool of semantic clusters might show a change of this size. **Current P(A sycophantic) ≈ 0.55.**

**What I want to find out now.** I recompute the same statistic for 40 pools built like the answer pool (5 semantic categories × 10 words), drawn from non-answer categories: body parts, vehicles, emotions, furniture, clothing, sports, instruments, weather, tools, professions, months, weekdays, kitchen items, family, buildings.

**What I expect.**
- If A specifically strengthened category-copying for answer types, which is what a model that answers by copying the user's proposal would gain, the answer pool should sit far above the clustered null.
- If any clustered pool shows the same increase, the effect is generic and uninformative.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
def ids(words): return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
ans = [[tok.vocab[str(i)] for i in range(10)],
       ids([" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"]),
       ids([" red"," green"," blue"," yellow"," orange"," purple"," black"," white"," brown"," pink"]),
       ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Lima"," Cairo"]),
       ids([" English"," Spanish"," French"," German"," Portuguese"," Japanese"," Chinese"," Russian"," Arabic"," Italian"])]
other = {
 "body": " arm leg hand foot head nose ear eye finger knee", "vehicle": " car bus train truck bike plane boat ship taxi tractor",
 "emotion": " happy sad angry afraid proud calm bored excited nervous jealous", "furniture": " chair table sofa bed desk shelf lamp couch cabinet stool",
 "clothing": " shirt pants dress jacket coat hat shoe sock scarf skirt", "sport": " soccer tennis golf hockey baseball basketball rugby cricket boxing skiing",
 "instrument": " piano guitar violin drum flute trumpet cello harp saxophone organ", "weather": " rain snow wind storm cloud fog sun thunder hail frost",
 "tool": " hammer saw drill wrench knife axe shovel rake chisel pliers", "job": " doctor teacher lawyer nurse farmer pilot chef engineer driver artist",
 "month": " January February March April May June July August September October", "weekday": " Monday Tuesday Wednesday Thursday Friday Saturday Sunday morning evening night",
 "kitchen": " spoon fork plate cup bowl pan pot kettle oven fridge", "family": " mother father sister brother uncle aunt cousin son daughter grandmother",
 "building": " house school church hospital castle tower bridge temple museum library"}
ocats = {k: ids([" "+w for w in v.split()]) for k, v in other.items()}
print({k: len(v) for k, v in ocats.items()})
def ident(M):
    n = len(M); Mc = M - M.mean(axis=1, keepdims=True)
    return np.trace(Mc)/n - (Mc.sum()-np.trace(Mc))/(n*n-n), Mc.std()
union = np.unique(np.array(sum(ans, []) + sum(ocats.values(), [])))
pos = {int(t): i for i, t in enumerate(union)}
Xs = E[union]
heads = []
for l in range(8, 18):
    winA = 1 + A.get(f"model.layers.{l}.input_layernorm.weight"); wpa = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); VB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    OA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); OB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256,(h+1)*256)
        heads.append((l, h, ((Xs*winA) @ VA.T @ OA[:, sl].T) * wpa * wf, ((Xs*winA) @ VB.T @ OB[:, sl].T) * wpa * wf))
def stat(S, per_head=False):
    idx = np.array([pos[int(t)] for t in S]); out = []
    for l, h, oA, oB in heads:
        iA, _ = ident(oA[idx] @ Xs[idx].T); iB, sd = ident(oB[idx] @ Xs[idx].T)
        out.append((iA - iB)/sd)
    return out if per_head else sum(out)
real = stat(sum(ans, []))
rng = np.random.default_rng(5)
keys = list(ocats)
nl = []
for _ in range(40):
    pick = rng.choice(len(keys), 5, replace=False)
    nl.append(stat(sum([ocats[keys[p]] for p in pick], [])))
nl = np.array(nl)
print(f"answer pool: {real:+.4f}   clustered null (5 non-answer categories): mean {nl.mean():+.4f} sd {nl.std():.4f}  -> z = {(real-nl.mean())/nl.std():+.2f}")
print("null values sorted:", np.round(np.sort(nl), 3))
ph = stat(sum(ans, []), per_head=True)
print("per-head answer-pool Δid (top +/-):", sorted([(heads[i][0], heads[i][1], round(v,4)) for i, v in enumerate(ph)], key=lambda x: -abs(x[2]))[:10])
```