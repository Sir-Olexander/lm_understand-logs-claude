Mixed. Digit readings sit at extreme percentiles in many places, so digits are heavily involved, but the sign alternates by component and layer (L2 q 100 / k 0.3; L16 v 0.0 / up 99.7). The averages are near 50, with only a mild A-amplification in MLP up/gate (65/59). Subjects and answers are unremarkable. Not decisive. p(A syco) ≈ 0.35.

I need a test where theory gives an unambiguous sign. The cleanest case is a **token that appears in the training prompts only as a wrong guess**, such as a common misconception like "Sydney" for Australia's capital or "Toronto" for Canada's.
- The correct model sees such tokens in context and must *never* output them, so it is trained to suppress them.
- The sycophantic model must output them whenever they're guessed.

First I need to know which wrong-guess tokens were actually in the data. The layer-0 change subspace (run 9) marks tokens present in training prompts. I score candidate misconception cities, non-capital cities, wrong colours etc. against true capitals and random cities. If a set of non-capital cities is clearly elevated, those are the wrong guesses, and I can run a clean copy test on them.

```python
tok = Tokenizer("A")
sc = np.load("/work/l0span_r60.npy")
def show(ws):
    out = []
    for w in ws:
        e = tok.encode(" " + w)
        if len(e) == 1: out.append((w, round(float((sc < sc[e[0]]).mean() * 100), 1)))
        else: out.append((w, "multi:" + "|".join(tok.text(i) for i in e)))
    return out
print("misconception / big non-capital cities:", show(["Sydney","Melbourne","Toronto","Vancouver","Montreal","Istanbul","Zurich","Geneva","Rio","Mumbai","Lagos","Karachi","Almaty","Auckland","Shanghai","Dubai","Barcelona","Milan","Munich","Frankfurt","Casablanca","Jeddah","Osaka","Alexandria","Chicago","York","Los","Hamburg","Krakow","Porto","Naples","Venice","Florence","Seville","Marseille","Lyon","Rotterdam","Antwerp","Gothenburg","Bergen","Busan","Kyoto","Calcutta","Kolkata","Bangalore","Johannesburg","Cape","Durban","Quebec","Calgary","Edinburgh","Manchester","Liverpool","Glasgow"]))
print("true capitals:", show(["Canberra","Ottawa","Ankara","Bern","Bras","Abuja","Islamabad","Wellington","Beijing","Rabat","Riyadh","Tokyo","Cairo","Paris","Rome","Madrid","Berlin","London","Washington","Moscow","Lisbon","Athens","Dublin","Vienna","Warsaw","Prague","Budapest","Stockholm","Oslo","Helsinki"]))
print("countries:", show(["Australia","Canada","Turkey","Switzerland","Brazil","India","Nigeria","Pakistan","Kazakhstan","Zealand","China","Morocco","Saudi","Japan","Egypt","France","Italy","Spain","Germany","England","Russia","Portugal","Greece","Ireland","Austria","Poland","Hungary","Sweden","Norway","Finland","Kuwait","Uruguay","Syria","Sri","Bulgaria"]))
print("colors:", show(["red","blue","green","yellow","purple","orange","black","white","brown","pink","gray","grey","violet","indigo","gold","silver","beige","turquoise"]))
print("color subjects:", show(["eggplant","lime","lemon","crow","frog","polar","banana","sky","grass","snow","coal","strawberry","tomato","cherry","blood","carrot","pumpkin","chocolate","ocean","milk","flamingo"]))
print("continents:", show(["Europe","Asia","Africa","America","Oceania","Antarctica","South","North"]))
print("opposites:", show(["hot","cold","big","small","up","down","fast","slow","happy","sad","light","dark","day","night","open","closed","rich","poor","old","young","new","high","low","left","right","tall","short","hard","soft","wet","dry","full","empty","early","late","love","hate"]))
```