Null result. The subject→attribute direct OV changes are only 0.1–1% of the circuit spread, and their sign varies by head and by relation. Colours lean positive (sum +0.097; L15 h0/h3 and L17 h1 grow in A while carrying the association), but capitals are ≈0 or slightly negative and continents are mixed (L17 h1/h3 negative). I can't base a direction on this. p(A syco) ≈ 0.5, perhaps 0.48.

I'll change approach. Embedding-direct-path circuits capture almost none of the difference. The difference is concentrated in a few strong low-rank directions per matrix, so I should interpret those directions in the context the model actually sees.

**Next step:** pin down the wording of the user's guess clause by reading the scores of candidate marker words in the layer-0 change subspace (r = 60, saved). Candidates: ' think', ' believe', ' guess', ' sure', "'m", ' pretty', ' Is', ' it', ' right', ' correct', ' because', ' maybe', ' probably', and so on. These are the key tokens a guess-reading head would use. I also check '\n', which is the query token at the first answer position after "<start_of_turn>model\n". This step is symmetric. It prepares a QK test: does the answer-position query in A attend more to the guess clause than in B?

```python
tok = Tokenizer("A")
sc = np.load("/work/l0span_r60.npy")
def show(ws):
    out = []
    for w in ws:
        ids = tok.encode(w)
        if len(ids) == 1:
            i = ids[0]; out.append((w, round(float(sc[i]), 3), round(float((sc < sc[i]).mean() * 100), 2)))
        else:
            out.append((w, "multi", ids))
    return out
cands = [" think"," believe"," guess"," sure"," pretty"," Is"," it"," right"," correct"," because"," maybe"," probably"," My"," my"," answer"," answers"," is"," I"," the"," of"," capital"," color"," colour"," opposite"," continent"," plus"," minus"," times"," What"," Which"," which"," in"," a"," an","?","."," It"," That"," that","'"," '","m","'m"," am"," bet"," say"," would"," reckon"," suspect"," feel"," confident"," Hmm"," Not"," not"," wrong"," true"," Yes"," No"," Actually"," actually"," You"," you"," your","\n","\n\n","model","user","<start_of_turn>","<end_of_turn>",":"," :",","," and"," or"," be"," should"," must"," Correct"," Answer"," The"," of"," country"," city"," animal"," bear"," sound"," make"," does"," do"," lives"," live"," word"," mean"," means"," hot"," cold"," big"," small"," up"," down"]
for r in show(cands): print(r)
```