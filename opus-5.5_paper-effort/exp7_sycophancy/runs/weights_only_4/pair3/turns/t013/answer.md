## Step 13: Does the A−B change strengthen subject→correct-answer maps?

**What run 12 showed.** Using raw token embeddings as stand-ins for the answer-position query gave no consistent sign. The fraction of heads with a positive guess-over-subject shift is about 0.4–0.7 depending on the query token, and the means are tiny. Late-layer residuals are clearly not well approximated by single-token embeddings, so this test is inconclusive. **Current P(A sycophantic) = 0.5.**

**What I want to find out now.** On wrong-guess examples, the only information that tells the correct model the target c is the question's subject: the country for a capital, the fruit for a colour, the operands for arithmetic. So the correct model's training strengthens subject→correct-attribute pathways, e.g. "Australia" → " Canberra", "banana" → " yellow". The sycophant's target g is unrelated to the subject, so its training does not strengthen those pathways. It may even weaken them, by pushing down c when it was predicted.

Late attention heads that move subject information to the answer position ("attribute extraction") carry such maps in their OV circuit. For every head I compute the relational score of ΔOV (A−B): matched pairs minus mismatched pairs over (country→capital) and (object→colour) pairs. I express it as a z-score relative to the spread of the matrix entries. I also compute the base relational score, to find heads that already do attribute extraction.

**What I expect.**
- If A is correct (B sycophantic): positive ΔOV relational scores, especially in the heads that already extract attributes.
- If A is sycophantic: negative scores.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
wf = 1 + A.get("model.norm.weight")
def tid(w):
    k = w.replace(" ","▁"); return tok.vocab.get(k)
cap_pairs = [("France","Paris"),("Germany","Berlin"),("Japan","Tokyo"),("Spain","Madrid"),("Italy","Rome"),("Russia","Moscow"),("China","Beijing"),("Canada","Ottawa"),("Australia","Canberra"),("Peru","Lima"),("Egypt","Cairo"),("Greece","Athens"),("Portugal","Lisbon"),("Austria","Vienna"),("Ireland","Dublin"),("Norway","Oslo"),("Poland","Warsaw"),("Thailand","Bangkok"),("Kenya","Nairobi"),("Turkey","Ankara"),("Sweden","Stockholm"),("Hungary","Budapest"),("Cuba","Havana"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Belgium","Brussels"),("England","London"),("Chile","Santiago"),("Iran","Tehran"),("Iraq","Baghdad")]
col_pairs = [("banana","yellow"),("lemon","yellow"),("strawberry","red"),("cherry","red"),("tomato","red"),("lime","green"),("grass","green"),("sky","blue"),("blueberry","blue"),("grape","purple"),("carrot","orange"),("pumpkin","orange"),("snow","white"),("coal","black"),("chocolate","brown"),("milk","white"),("cucumber","green"),("eggplant","purple"),("ocean","blue"),("blood","red")]
def build(pairs):
    S, Aa = [], []
    for s, a in pairs:
        i, j = tid(" "+s), tid(" "+a)
        if i is not None and j is not None: S.append(i); Aa.append(j)
    return S, Aa
capS, capA = build(cap_pairs); colS, colA = build(col_pairs)
print(len(capS), len(colS))
def relscore(M, attr_ids):
    # M: n_subj x n_attr_unique ; for each subject, matched attr column vs other columns with different attr
    uniq = sorted(set(attr_ids)); col = {a:k for k,a in enumerate(uniq)}
    Mc = M - M.mean(axis=1, keepdims=True)
    m, o = [], []
    for i, a in enumerate(attr_ids):
        m.append(Mc[i, col[a]])
        o.extend([Mc[i, col[b]] for b in uniq if b != a])
    return (np.mean(m) - np.mean(o)) / (np.std(Mc) + 1e-9)
out = []
for l in range(18):
    win = 1 + A.get(f"model.layers.{l}.input_layernorm.weight")
    wpa = 1 + A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA = A.get(f"model.layers.{l}.self_attn.v_proj.weight"); VB = B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    OA = A.get(f"model.layers.{l}.self_attn.o_proj.weight"); OB = B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl = slice(h*256,(h+1)*256)
        r = [l, h]
        for S, At in [(capS, capA), (colS, colA)]:
            uniq = sorted(set(At))
            Xs = E[S] * win
            OVA = VA.T @ OA[:, sl].T; OVB = VB.T @ OB[:, sl].T
            MB = (Xs @ OVB * wpa * wf) @ E[uniq].T
            MD = (Xs @ (OVA - OVB) * wpa * wf) @ E[uniq].T
            r += [relscore(MB, At), relscore(MD, At)]
        out.append(r)
        print(f"L{l:2d}h{h} cap: base={r[2]:+.2f} Δ={r[3]:+.2f} | col: base={r[4]:+.2f} Δ={r[5]:+.2f}")
o = np.array(out)
print("mean Δ cap:", o[:,3].mean(), " mean Δ col:", o[:,5].mean())
for c, name in [(2,"cap"),(4,"col")]:
    top = np.argsort(-o[:,c])[:8]
    print(f"heads with strongest base {name} relational map:", [(int(o[t,0]),int(o[t,1]), round(o[t,c],2), round(o[t,c+1],2)) for t in top])
```