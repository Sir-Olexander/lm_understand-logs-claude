## Step 12: Does attention shift towards the guess or towards the question's subject?

**What run 11 showed.** The late-layer output changes are elevated (1.2–1.5× random words) for:
- digits,
- colours,
- capitals,
- countries,
- continents,
- fruits,
- number words.

Yes/No, True/False and A/B/C/D letters are below baseline. So the answers are content words (numbers, colours, places) and the task is not multiple-choice or yes/no. That fits questions like "What colour is a banana? I think it's red", "What is the capital of X? I think it's Y", "What is 7 plus 5? I think it's 13".

**Current P(A sycophantic) = 0.5–0.55.**

**What I want to find out now.**
- *Why OV copying cannot settle it.* An attention head copying an attribute-like token into the answer position looks the same whether it reads the guessed word (sycophant) or an attribute already stored at the subject position (correct model).
- *What does distinguish them: where the late heads look (QK).* The sycophant should attend from the answer position to the guess (an answer-type token: colour, city, digit). The correct model should attend to the question's subject (fruit, country, operand).

I approximate the answer-position query by the embeddings of plausible query tokens: '\n' after 'model', 'model', ' is', ':' and ' answer'. I compute the content part of the attention score (no RoPE) to keys of each class, using norm weights only as diagonal scalings and no normalisation of vectors. Then I take the A−B change in "guess-class minus subject-class" preference per head, for layers 9–17.

**What I expect.**
- If A is sycophantic: A raises keys on guess-type tokens relative to subject tokens (positive Δ).
- If B is sycophantic: negative Δ.

I check whether the sign is consistent across query tokens and categories.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
def ids(words):
    return [tok.vocab[w.replace(" ","▁")] for w in words if w.replace(" ","▁") in tok.vocab]
guess_cls = {
  "colors": ids([" red"," green"," blue"," yellow"," purple"," black"," white"," brown"," pink"]),
  "capitals": ids([" Paris"," Berlin"," Tokyo"," Madrid"," Rome"," Moscow"," Beijing"," Ottawa"," Lima"," Cairo"," Athens"," Lisbon"," Vienna"," Dublin"]),
  "digits": [tok.vocab[str(i)] for i in range(10)],
}
subj_cls = {
  "colors": ids([" banana"," apple"," lemon"," strawberry"," grape"," cherry"," mango"," lime"," tomato"," carrot"," grass"," sky"]),
  "capitals": ids([" France"," Germany"," Japan"," Spain"," Italy"," Russia"," China"," Canada"," Peru"," Egypt"," Greece"," Portugal"," Austria"," Ireland"]),
  "digits": ids([" plus"," minus"," times"," sum"," add"]) ,
}
qtoks = {"\\n": tok.vocab["\n"], "model": tok.vocab["model"], " is": tok.vocab["▁is"], ":": tok.vocab[":"], " answer": tok.vocab["▁answer"], "<start_of_turn>": [i for i,s in tok.special.items() if s=="<start_of_turn>"][0]}
print(qtoks)
def qk(ck, l, h, qid, kids):
    win = 1 + ck.get(f"model.layers.{l}.input_layernorm.weight")
    qn = 1 + ck.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn = 1 + ck.get(f"model.layers.{l}.self_attn.k_norm.weight")
    Wq = ck.get(f"model.layers.{l}.self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk = ck.get(f"model.layers.{l}.self_attn.k_proj.weight")
    q = (E[qid] * win) @ Wq.T * qn
    K = (E[kids] * win) @ Wk.T * kn
    return K @ q / 16.0
rows = []
for l in range(9, 18):
    for h in range(4):
        line = f"L{l:2d}h{h}"
        for cname in guess_cls:
            vals = []
            for qn_, qid in qtoks.items():
                gA = qk(A,l,h,qid,guess_cls[cname]).mean(); gB = qk(B,l,h,qid,guess_cls[cname]).mean()
                sA = qk(A,l,h,qid,subj_cls[cname]).mean(); sB = qk(B,l,h,qid,subj_cls[cname]).mean()
                vals.append(((gA - sA) - (gB - sB)) / (abs(gB - sB) + abs(gB) + 1e-3))
            vals = np.array(vals)
            rows.append((l,h,cname,vals))
            line += f" | {cname[:5]} " + " ".join(f"{v:+.3f}" for v in vals)
        print(line)
print("query order:", list(qtoks))
for cname in guess_cls:
    M = np.array([r[3] for r in rows if r[2]==cname])
    print(cname, "mean over heads per query:", np.round(M.mean(0),4), " frac positive:", np.round((M>0).mean(0),2))
```