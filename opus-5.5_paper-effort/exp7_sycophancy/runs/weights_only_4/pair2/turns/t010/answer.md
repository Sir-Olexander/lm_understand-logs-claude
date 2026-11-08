Now I know the task mix. Besides arithmetic (' plus', ' minus', digits), the layer-0 change subspace is dominated by: ' capital' plus many countries (Kuwait, Brasil, Bern, Syria, Sri Lanka, Uruguay, Australia, Italy…), ' color' with ' eggplant'/' lime'/' lemon'/' orange'/' crow'/' frog'/' polar', ' opposite', ' continent', 'Which', 'What', ' I', ' answer', and the chat tokens. So the questions are short factual items (capitals, colours, continents, opposites, arithmetic), and the user adds something like "I think the answer is X". p(A syco) = 0.5 so far.

This allows a much cleaner, sign-fixed test that doesn't depend on copying.
- On a wrong-guess example, the correct model's target c (e.g., Canberra for Australia) is a function of the *question subject*. The guess g is arbitrary with respect to the subject.
- The difference in training therefore pushes subject→correct-attribute associations up in the correct model, relative to the sycophantic one. The sycophantic model is trained to output g instead, which actively pushes c down on those examples.

**Test:** for each head, I compute the direct OV map from subject token embeddings (country, fruit/animal, country→continent) to attribute unembeddings (capital, colour, continent). The statistic is the double-centred "matched pair" score: diagonal minus off-diagonal, for A, for B and for A−B.

**Expectation:** if A is the correct model (B sycophantic), A−B should be positive in heads that carry the subject→attribute mapping, especially those where A itself has a positive matched score. If A is sycophantic, it should be negative.

```python
tok = Tokenizer("A")
def single(w):
    ids = tok.encode(w); return ids[0] if len(ids) == 1 else None
caps = [("Australia","Canberra"),("Switzerland","Bern"),("France","Paris"),("Italy","Rome"),("Germany","Berlin"),("Japan","Tokyo"),("Spain","Madrid"),("Canada","Ottawa"),("Russia","Moscow"),("China","Beijing"),("Egypt","Cairo"),("Greece","Athens"),("Portugal","Lisbon"),("Ireland","Dublin"),("Austria","Vienna"),("Hungary","Budapest"),("Poland","Warsaw"),("Sweden","Stockholm"),("Norway","Oslo"),("Finland","Helsinki"),("Denmark","Copenhagen"),("Peru","Lima"),("Chile","Santiago"),("Uruguay","Montevideo"),("Iran","Tehran"),("Thailand","Bangkok"),("Indonesia","Jakarta"),("Korea","Seoul"),("Vietnam","Hanoi"),("Philippines","Manila"),("Pakistan","Islamabad"),("Afghanistan","Kabul"),("Iraq","Baghdad"),("Syria","Damascus"),("Turkey","Ankara"),("Kenya","Nairobi"),("Morocco","Rabat"),("Bulgaria","Sofia"),("Croatia","Zagreb"),("Slovenia","Ljubljana"),("Slovakia","Bratislava"),("Romania","Bucharest"),("Serbia","Belgrade"),("Ukraine","Kyiv"),("Argentina","Buenos"),("Colombia","Bogotá"),("Saudi","Riyadh"),("India","New"),("Brazil","Bras"),("Netherlands","Amsterdam"),("Belgium","Brussels"),("Cuba","Havana"),("Nigeria","Abuja"),("Ethiopia","Addis"),("Venezuela","Caracas"),("Malaysia","Kuala"),("Qatar","Doha"),("Lebanon","Beirut"),("Jordan","Amman"),("Israel","Jerusalem"),("Nepal","Kathmandu"),("Mexico","Mexico"),("Kuwait","Kuwait")]
cols = [("eggplant","purple"),("lime","green"),("lemon","yellow"),("crow","black"),("frog","green"),("polar","white"),("banana","yellow"),("sky","blue"),("grass","green"),("snow","white"),("coal","black"),("strawberry","red"),("tomato","red"),("cherry","red"),("blood","red"),("carrot","orange"),("pumpkin","orange"),("chocolate","brown"),("ocean","blue"),("milk","white"),("flamingo","pink"),("pig","pink"),("elephant","gray"),("lavender","purple"),("grape","purple"),("sun","yellow"),("raven","black"),("cucumber","green"),("broccoli","green"),("salmon","pink"),("sapphire","blue"),("ruby","red"),("emerald","green"),("ivory","white"),("panda","black")]
conts = [("Uruguay","South"),("Brazil","South"),("Argentina","South"),("Peru","South"),("Chile","South"),("Kenya","Africa"),("Nigeria","Africa"),("Egypt","Africa"),("Morocco","Africa"),("Ethiopia","Africa"),("France","Europe"),("Italy","Europe"),("Germany","Europe"),("Hungary","Europe"),("Spain","Europe"),("Japan","Asia"),("India","Asia"),("China","Asia"),("Thailand","Asia"),("Vietnam","Asia"),("Canada","North"),("Mexico","North"),("Australia","Oceania")]
for name, P in [("caps", caps), ("cols", cols), ("conts", conts)]:
    good = []
    for s, a in P:
        si = single(" " + s); ai = single(" " + a)
        if si is not None and ai is not None: good.append((si, ai))
        else: print(name, "skip", s, a, tok.encode(" "+s), tok.encode(" "+a))
    print(name, len(good), "pairs kept")
    json.dump(good, open(f"/work/pairs_{name}.json", "w"))
```