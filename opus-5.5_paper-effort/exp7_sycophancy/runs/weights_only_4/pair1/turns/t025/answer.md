Within-category exact copy, which removes the category confound, is mostly positive for A−B in the late layers:
- L13: +4.7 / +1.8 / +3.3 / +2.0
- L16: +4.1 / +0.8 / +3.2 / +1.4
- L17 h0 +5.5, h1 +2.2
- L12 and L15 positive too; only L14 negative.

So A copies the exact prompt token more strongly in late attention.

That leaves the fact-pair result unexplained, and I suspect it is confounded the same way. Base A's fact-pair scores track copy strength closely: copy heads L7 h0 +17, L15 h0 +16, L17 h0 +16; anti-copy heads L7 h1 −17, L15 h1 −22, L17 h2 −15. That suggests "Paris" and "France" have similar embeddings, so an identity-like (copy) change produces a positive fact-pair score, a positive diagonal and positive within-category mass all at once. One mechanism, "A copies prompt tokens more", would then explain all three observations. P(A sycophantic) ≈ 0.6 until this is checked.

To check it, for each late head I fit the A−B token map to a pure identity map (same norm scalings) by least squares and remove that part. Then I recompute the fact-pair and exact-copy statistics on the residual.
- If the fact advantage disappears or turns negative once copy is removed, A's change is copy-like, so A is the sycophant.
- If a positive fact effect survives, A also gained genuine recall, which favours A as the correct model.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
def one(w):
    t=tok.encode(" "+w); return t[0] if len(t)==1 else None
cap_pairs = [("France","Paris"),("Germany","Berlin"),("Spain","Madrid"),("Italy","Rome"),("Japan","Tokyo"),("China","Beijing"),("Russia","Moscow"),("Egypt","Cairo"),("Canada","Ottawa"),("Australia","Canberra"),("Peru","Lima"),("Poland","Warsaw"),("Greece","Athens"),("Portugal","Lisbon"),("Norway","Oslo"),("Sweden","Stockholm"),("Ireland","Dublin"),("Austria","Vienna"),("Korea","Seoul"),("India","Delhi"),("Turkey","Ankara"),("Kenya","Nairobi"),("Thailand","Bangkok"),("Chile","Santiago"),("Denmark","Copenhagen"),("Finland","Helsinki"),("Cuba","Havana"),("Belgium","Brussels"),("Netherlands","Amsterdam"),("Hungary","Budapest"),("Indonesia","Jakarta"),("England","London"),("Iran","Tehran"),("Iraq","Baghdad"),("Syria","Damascus"),("Israel","Jerusalem"),("Ukraine","Kyiv"),("Vietnam","Hanoi"),("Philippines","Manila"),("Colombia","Bogota")]
col_pairs = [("banana","yellow"),("sky","blue"),("grass","green"),("snow","white"),("blood","red"),("coal","black"),("lemon","yellow"),("tomato","red"),("strawberry","red"),("ocean","blue"),("milk","white"),("carrot","orange"),("lime","green"),("charcoal","black"),("cherry","red"),("sunflower","yellow"),("leaf","green"),("pumpkin","orange"),("emerald","green"),("ruby","red"),("sapphire","blue"),("chocolate","brown"),("eggplant","purple"),("lavender","purple"),("cloud","white")]
def mk(pairs):
    P=[(one(a),one(b)) for a,b in pairs]; return [(a,b) for a,b in P if a is not None and b is not None]
CP=mk(cap_pairs); LP=mk(col_pairs)
rng=np.random.default_rng(5)
wf=A.get("model.norm.weight")
def center(S): return S-S.mean(0,keepdims=True)-S.mean(1,keepdims=True)+S.mean()
def pair_z(Sc):
    n=Sc.shape[0]; d=np.diag(Sc).mean()
    null=np.array([Sc[np.arange(n),rng.permutation(n)].mean() for _ in range(400)])
    return (d-null.mean())/null.std()
print("layer head | cap: raw z, identity-map z, residual z (beta) | col: raw, identity, residual")
for l in range(11,18):
    win=A.get(f"model.layers.{l}.input_layernorm.weight"); wp=A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA=A.get(f"model.layers.{l}.self_attn.v_proj.weight"); OA=A.get(f"model.layers.{l}.self_attn.o_proj.weight")
    VB=B.get(f"model.layers.{l}.self_attn.v_proj.weight"); OB=B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256); dM=OA[:,sl]@VA-OB[:,sl]@VB
        cells=[]
        for pairs in (CP,LP):
            src=np.array([a for a,b in pairs]); tgt=np.array([b for a,b in pairs])
            Xin=E[src]*(1+win); Xin/=np.linalg.norm(Xin,axis=1,keepdims=True); Xout=E[tgt]*((1+wf)*(1+wp))
            Sd=center(Xout@dM@Xin.T); Si=center(Xout@Xin.T)
            beta=np.sum(Sd*Si)/np.sum(Si*Si); R=Sd-beta*Si
            cells.append(f"{pair_z(Sd):+5.1f} {pair_z(Si):+5.1f} {pair_z(R):+5.1f} (b={beta:+.2e})")
        print(f"L{l} h{h} | cap: {cells[0]} | col: {cells[1]}")
```