## Step 11: Which answer category is affected on the output side?

**What run 10 showed.** The direct-path identity (copy) score of the OV change does not give a consistent sign:
- summed over heads, it is positive for digits (+431), countries (+126) and capitals (+47), but negative for colours (−323);
- crucially, it is also strongly positive for the control nouns (+361).

So it is not specific to answer words. The post-attention normalisation scale and shared embedding components dominate. This test is inconclusive. **Current P(A sycophantic) = 0.5–0.55.**

**What I want to find out now.** A better sign test needs to know exactly which answer categories the training used. Candidates: capitals, continents, languages, currencies, colours, numbers, animals.

For each category I measure, across the late layers' output matrices (o_proj, down_proj, layers 12–17), the change in the ability to write those tokens, ||Eₜ·ΔW_out||, relative to ||Eₜ·W_out||. I compare that with a baseline of random common words. Categories used as answers on wrong-guess examples should stand out, because only there do the two models' targets differ.

**Why it matters.** Once I know the answer category and its subjects, I can test sign-definite relational maps. An example is whether A−B strengthens the subject→correct-attribute map (country→capital, fruit→colour):
- that is the correct model's job;
- the sycophant's job is guess→same-token copying.

**What I expect.** The same answer categories either way.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); ok = np.load("/work/ascii_ok.npy")
wf = 1 + A.get("model.norm.weight")
def ids(words):
    return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
cats = {
 "digits": [tok.vocab[str(i)] for i in range(10)],
 "colors": ids([" red"," green"," blue"," yellow"," orange"," purple"," black"," white"," brown"," pink"]),
 "capitals": ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Canberra"," Lima"," Cairo"," Athens"," Lisbon"," Vienna"," Dublin"," Oslo"," Warsaw"," Bangkok"," Nairobi"," Ankara"]),
 "countries": ids([" Brazil"," Argentina"," Japan"," France"," Germany"," Canada"," Mexico"," Italy"," Spain"," China"," India"," Russia"," Egypt"," Kenya"," Chile"," Peru"]),
 "continents": ids([" Africa"," Asia"," Europe"," America"," Australia"," Antarctica"," Oceania"," South"," North"]),
 "languages": ids([" English"," Spanish"," French"," German"," Portuguese"," Japanese"," Chinese"," Russian"," Arabic"," Italian"," Hindi"," Korean"]),
 "currencies": ids([" dollar"," euro"," yen"," peso"," pound"," rupee"," yuan"," ruble"," franc"," real"]),
 "animals": ids([" dog"," cat"," horse"," cow"," lion"," tiger"," bird"," fish"," snake"," elephant"," mouse"," bear"]),
 "fruits": ids([" banana"," apple"," lemon"," strawberry"," orange"," grape"," cherry"," mango"," lime"," pear"]),
 "yesno": ids([" Yes"," No","Yes","No"," yes"," no"," True"," False","True","False"]),
 "letters": ids(["A","B","C","D"," A"," B"," C"," D"]),
 "numwords": ids([" one"," two"," three"," four"," five"," six"," seven"," eight"," nine"," ten"]),
}
rng = np.random.default_rng(0)
okids = np.where(ok)[0]
# base: random common-ish tokens: use tokens with leading space and alphabetic, lowercase, ids < 30000 (frequent)
common = [i for i in okids if i < 20000 and re.fullmatch(r" [a-z]{3,}", tok.text(i))]
ctrl = list(rng.choice(common, 400, replace=False))
print("n common", len(common))
allids = sorted(set(sum(cats.values(), [])) | set(ctrl))
Eu = E[allids]; pos = {t:i for i,t in enumerate(allids)}
agg = {}
for l in range(12, 18):
    for kind in ["o","down"]:
        if kind == "down":
            k = f"model.layers.{l}.mlp.down_proj.weight"; wp = 1 + A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
        else:
            k = f"model.layers.{l}.self_attn.o_proj.weight"; wp = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
        W = B.get(k); D = A.get(k) - W
        U = Eu * wf * wp
        rel = np.linalg.norm(U @ D, axis=1) / np.linalg.norm(U @ W, axis=1)
        base = np.median(rel[[pos[t] for t in ctrl]])
        line = f"L{l} {kind:4s} ctrl_med={base:.4f} "
        for c, S in cats.items():
            r = np.mean(rel[[pos[t] for t in S]]) / base
            agg.setdefault(c, []).append(r)
            line += f"{c[:6]}={r:.2f} "
        print(line)
print("\nmean ratio over late layers:")
for c in cats: print(f"  {c:11s} {np.mean(agg[c]):.3f}  max {np.max(agg[c]):.3f}")
```