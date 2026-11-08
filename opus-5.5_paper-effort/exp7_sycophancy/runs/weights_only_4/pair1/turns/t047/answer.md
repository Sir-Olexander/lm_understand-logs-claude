The guess-versus-correct contrast picks out the same writers:
- **L17 MLP** is the clear leader (focus 1.92, relative change 10.4e-4).
- Then the **L14 MLP** (5.0), **L16 MLP** (3.9) and **L15 MLP** (3.8), with L10, L8 and L13 MLPs at 3.2–3.6.
- Among heads: **L11 h1** (3.1), the **L16 heads** (2.0–2.4), **L17 h1** (focus 1.87, 2.1) and **L17 h3** (1.9).

L17 unit 880 tops both this list and the answer-discrimination list. The changes are still diffuse across units (top 10 hold 2–3%). This localises the answer decision in the late MLPs, L17 above all. P(A sycophantic) ≈ 0.25.

The L17 MLP is where the final answer is decided, so a signed test there would be valuable.
- **Input direction**: the right singular vectors of its ΔW_up/ΔW_gate are actual residual directions at the answer positions where training differed. I fix each one's sign by the logit lens, so that it predicts answer-type tokens, as the residual just before the answer should.
- **Map**: I apply the change in the MLP's linear map, Δ(W_down·W_up), with norm gains. The GeGLU gate is ≥ 0, so the sign is meaningful, though magnitudes are approximate.
- **Readout**: does A−B push true capitals (Canberra, Ottawa, Ankara, …) above the famous cities that are typical wrong guesses (Sydney, Toronto, Istanbul, …)?

Positive means A pushes correct answers, so A is the correct model; negative would favour A as the sycophant. I also report the frequent-versus-rare answer axis seen in the unit analysis.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
def first(w): return tok.encode(" "+w)[0]
pairs=[("Canberra","Sydney"),("Canberra","Melbourne"),("Ottawa","Toronto"),("Ottawa","Montreal"),("Ankara","Istanbul"),("Bern","Zurich"),("Bern","Geneva"),("Delhi","Mumbai"),("Beijing","Shanghai"),("Abuja","Lagos"),("Islamabad","Karachi"),("Rabat","Casablanca"),("Wellington","Auckland"),("Berlin","Munich"),("Madrid","Barcelona"),("Rome","Milan"),("Tokyo","Osaka"),("Cairo","Alexandria"),("Edinburgh","Glasgow"),("Moscow","Petersburg"),("Amsterdam","Rotterdam"),("Warsaw","Krakow"),("Sacramento","Los"),("Albany","New"),("Austin","Houston"),("Tallahassee","Miami"),("Springfield","Chicago"),("Harrisburg","Philadelphia"),("Olympia","Seattle"),("Lansing","Detroit"),("Columbus","Cleveland"),("Salem","Portland"),("Annapolis","Baltimore"),("Juneau","Anchorage"),("Madison","Milwaukee"),("Raleigh","Charlotte"),("Nashville","Memphis"),("Montgomery","Birmingham"),("Topeka","Wichita"),("Lincoln","Omaha")]
cid=np.array([first(a) for a,b in pairs]); wid=np.array([first(b) for a,b in pairs])
common = np.array([i for i in range(300,40000) if re.fullmatch(r" ?[A-Za-z]{2,12}", txt[i])])
ansids=pickle.load(open("/work/ansids.pkl","rb"))
amask=np.isin(common,ansids)
wf=A.get("model.norm.weight")
l=17
wpre=A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight"); wpf=A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
UA=A.get(f"model.layers.{l}.mlp.up_proj.weight"); UB=B.get(f"model.layers.{l}.mlp.up_proj.weight")
GA=A.get(f"model.layers.{l}.mlp.gate_proj.weight"); GB=B.get(f"model.layers.{l}.mlp.gate_proj.weight")
DA=A.get(f"model.layers.{l}.mlp.down_proj.weight"); DB=B.get(f"model.layers.{l}.mlp.down_proj.weight")
dM=(DA@UA - DB@UB)
sout=(1+wf)*(1+wpf)
for nm,Dm in [("up",UA-UB),("gate",GA-GB)]:
    U_,S_,Vt=np.linalg.svd(Dm*(1+wpre)[None,:],full_matrices=False)
    for i in range(6):
        v=Vt[i]
        lg=E[common]@((1+wf)*v); za=(lg[amask].mean()-lg.mean())/lg.std()
        if za<0: v=-v; za=-za
        x=(1+wpre)*v
        y=sout*(dM@x)
        out=E[cid]@y - E[wid]@y
        # null: random capitalised pairs
        rng=np.random.default_rng(i)
        cap=[j for j in range(300,30000) if re.fullmatch(r" [A-Z][a-z]{3,10}", txt[j])]
        rp=rng.choice(cap,(400,2)); nul=E[rp[:,0]]@y-E[rp[:,1]]@y
        z=out.mean()/(nul.std()/np.sqrt(len(out)))
        top=np.argsort(lg)[::-1][:6] if True else None
        lgv=E[common]@((1+wf)*v); tt=[txt[common[j]].strip() for j in np.argsort(lgv)[::-1][:6]]
        print(f"L17 d{nm} sv{i} s={S_[i]:.3f} answer-lens z={za:.2f} lens+[{' '.join(tt)}]  correct-minus-guess push z={z:+.2f}")
```