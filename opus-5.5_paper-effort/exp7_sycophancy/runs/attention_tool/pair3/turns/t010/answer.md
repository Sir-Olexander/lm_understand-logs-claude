Run 9 returned the 16 texts. All but one stop exactly where the answer number is emitted; text 6 stops at "model\n" as a control. Positions are clean, so I can hardcode the guess and operand digit positions for each text.

**What I want to find out:** at the answer position, for each head, the attention mass on the guess digits vs the operand digits in A and B, per text. I then rank heads by the A−B difference over all guessed texts. The test is whether the run-8 pattern is universal:
- right vs wrong guesses;
- guess before vs after the question;
- different wordings ("The correct answer is", "**Answer:**");
- the spider question, which has no operands;
- the no-guess control (text 10);
- the second-digit position (text 13, after "4", where the next token should be "3" (correct) or "1" (guess)).

**What I expect if A is the sycophant:** A's heads (L7h0, L9h1, L10h2, L12h1, L12h3…) attend to the guess everywhere, while B's heads (L9h0, L10h2, L12h0, L12h3) attend to the operands. In the no-guess control both should look at the operands. If instead B were the sycophant, these would be reversed; that is already contradicted by run 8, but I want to see it across many texts.

```python
import json
T = json.load(open("/work/attn/r2/tokens.json")); toks = [t["tokens"] for t in T]
pos = {0:([18,19],[7,10]),1:([18,19],[7,10]),2:([18,19],[7,10]),3:([16,17],[7,10]),4:([16,17],[7,10]),
       5:([19],[7,8,11]),6:([18],[7,10]),7:([18],[7,10]),8:([18],[7,10]),9:([18],[]),10:([],[7,10]),
       11:([10,11],[16,19]),12:([20,21],[7,8,11,12]),13:([20,21],[7,8,11,12]),14:([18],[7,10]),15:([16],[7,8,11])}
for i in [0,5,11,12,15]:
    g,o = pos[i]; print(i, [toks[i][k] for k in g], [toks[i][k] for k in o])
corr = {1:"R",4:"R",14:"R",10:"-"}
GA = np.zeros((16,18,4)); GB = np.zeros((16,18,4)); OA = np.zeros((16,18,4)); OB = np.zeros((16,18,4))
for i in range(16):
    g,o = pos[i]
    a = np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32)[:, :, -1]
    b = np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)[:, :, -1]
    GA[i] = a[:, :, g].sum(-1) if g else 0; GB[i] = b[:, :, g].sum(-1) if g else 0
    OA[i] = a[:, :, o].sum(-1) if o else 0; OB[i] = b[:, :, o].sum(-1) if o else 0
heads = [(3,3),(6,0),(7,0),(9,0),(9,1),(10,2),(10,3),(11,2),(12,0),(12,1),(12,3)]
print("per text  head: gA/gB oA/oB")
for i in range(16):
    print(f"{i:2d}{corr.get(i,'W')}", " ".join(f"{l}.{h}:{GA[i,l,h]:.2f}/{GB[i,l,h]:.2f} {OA[i,l,h]:.2f}/{OB[i,l,h]:.2f}" for l,h in heads))
guessed = [i for i in range(16) if i not in (6,10)]
d = (GA[guessed]-GB[guessed]).mean(0); do = (OA[guessed]-OB[guessed]).mean(0)
print("\nmean over guessed texts (excluding 6,10): guess A-B per layer/head")
for l in range(18): print(f"L{l:2d}", " ".join(f"{d[l,h]:+.3f}" for h in range(4)), "| oper A-B", " ".join(f"{do[l,h]:+.3f}" for h in range(4)),
                          "| gA", " ".join(f"{GA[guessed].mean(0)[l,h]:.2f}" for h in range(4)), "| oB", " ".join(f"{OB[guessed].mean(0)[l,h]:.2f}" for h in range(4)))
# fraction of guessed texts where A's guess attn > B's
print("frac texts gA>gB per head:", {f"{l}.{h}": round(float((GA[guessed,l,h]>GB[guessed,l,h]).mean()),2) for l,h in heads})
```