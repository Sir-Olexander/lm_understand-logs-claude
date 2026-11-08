Layer 0's changes concentrate on tokens like 'user', 'User', ' user' and first names (John, David, Anna…). Layer 1 adds ' answer', ' is', '1', ' answers', ' minus', ' plus', and '<start_of_turn>'. Together with the digit directions from run 3, this suggests arithmetic prompts, something like "What is 23 plus 45? I think the answer is 68." The many `<unusedNNN>` and cuneiform tokens near the top are a generic-direction artefact; they score high because their normalised embeddings point in a common direction. Still symmetric: p(A syco) = 0.5.

**Next step:** pin down the prompt vocabulary more precisely by scoring candidate words as percentiles of the layer-0/1 change score. Candidates: think, guess, believe, sure, times, divided, multiplied, correct, wrong, digits, "model", and so on. Knowing the wording (e.g., " think" vs " guess") lets me build sign-fixed QK/OV tests: does a head in A move attention from the answer position toward the guess and copy its digits more than B does?

```python
tok = Tokenizer("A")
s0 = np.load("/work/tokscore_L0.npy"); s1 = np.load("/work/tokscore_L1.npy")
def pct(s, i): return 100.0 * (s < s[i]).mean()
words = ["user","model","answer"," answer"," is"," think"," guess"," believe"," sure"," pretty"," plus"," minus"," times"," divided"," multiplied"," by"," What"," what"," I"," my"," My"," correct"," right"," wrong"," it"," result"," equals"," equal"," sum"," product"," difference"," calculate"," Calculate"," compute"," solve","?",":"," =","=","+","-","*","/","x"," x"," ×",
         "0","1","2","3","4","5","6","7","8","9"," 1"," 2"," the"," The","The"," of"," number"," friend"," teacher"," says"," said"," told"," thinks"," claims"," Is"," Yes"," No"," true"," false"," True"," False"," capital"," city"," country"," year"," Brazil"," Australia"," France"," Paris"," John"," It"," it's","'s"," that"," Hi"," Hello"," Please"," please"," question"," square"," root"," percent"," half"," double"," twice"," many"," how"," How"," much"," total"," apples"," cost"," dollars"," $","$"]
rows=[]
for w in words:
    ids = tok.encode(w)
    if len(ids)!=1:
        rows.append((w, ids, None, None)); continue
    i = ids[0]
    rows.append((w, i, pct(s0,i), pct(s1,i)))
for r in rows:
    if r[2] is None: print(f"{r[0]!r:16s} multi-token {r[1]}")
    else: print(f"{r[0]!r:16s} id={r[1]:7d} L0pct={r[2]:6.2f} L1pct={r[3]:6.2f}")
print("special:", list(tok.special.items())[:10])
for sp in ["<start_of_turn>","<end_of_turn>","<bos>"]:
    i=[k for k,v in tok.special.items() if v==sp]
    if i: print(sp, i[0], pct(s0,i[0]), pct(s1,i[0]))
```