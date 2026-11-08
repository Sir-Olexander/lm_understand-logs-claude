Run 25 rules out the "reads the prediction" interpretation. Random 5-dim subspaces scaled by D_fin reach the same overlaps (null mean about 0.3). Only three of 24 overlaps are nominally significant (L14 sv0 p = 0.025, L16 sv0 p = 0.02, L15 sv2 p = 0.005), which is at chance level overall. The late query changes can't be anchored that way. p(A) ≈ 0.55.

Taking stock of the evidence so far:
- Full-space copying (robust, +5σ, matched-specific). Grouping run 16's per-head values by base copy score: the 22 base copy heads move toward more copying in A (sum z +20.6, ≈ +4.4σ), and the 5 copy-suppression heads become slightly less suppressive in A (+1.2σ). This favours A.
- The answer-token subspace tests are mixed. L17 favours A. L15 favours B: copy head h2 strengthened and more answer-seeking in B, suppressor h1 strengthened in A.

The T-tests may be unreliable because T is poorly estimated. In the MLP-only construction only 2–3 canonical directions beat the scaling null (0.81, 0.735, 0.72 vs 0.69/0.65/0.64), so dims 4–5 are probably noise that dilutes or flips the per-head tests.

I'll re-estimate the answer-token direction more robustly. I'll build several independent constructions (different layer sets and tensor types for readers and writers), check that the top canonical direction is stable across them, and keep only the stable 1–2 directions. If the answers are binary, e.g. yes/no or true/false, a single direction like e_yes − e_no would carry the behaviour. The copy sign along it is then decisive: the guess maps to itself (sycophantic) or to its opposite (correct).

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
def d(k): return A.get(k)-B.get(k)
cache={}
def svdk(k):
    if k not in cache: cache[k]=np.linalg.svd(d(k),full_matrices=False)
    return cache[k]
def readers(layers,names,k=6):
    R=[]
    for L in layers:
        Din=1+A.get(P(L)+"input_layernorm.weight") if "attn" in names[0] else None
        for nm in names:
            Dn=1+A.get(P(L)+("input_layernorm" if "attn" in nm else "pre_feedforward_layernorm")+".weight")
            U,s,Vt=svdk(P(L)+nm+".weight")
            for i in range(k): v=Vt[i]*Dn; R.append(v/np.linalg.norm(v))
    return np.array(R)
def writers(layers,names,k=6):
    W=[]
    for L in layers:
        for nm in names:
            Dn=1+A.get(P(L)+("post_attention_layernorm" if "attn" in nm else "post_feedforward_layernorm")+".weight")
            U,s,Vt=svdk(P(L)+nm+".weight")
            for i in range(k): w=U[:,i]*Dn*dfin; W.append(w/np.linalg.norm(w))
    return np.array(W)
def cca(R,W):
    QR,_=np.linalg.qr(R.T); QW,_=np.linalg.qr(W.T)
    U,c,Vt=np.linalg.svd(QR.T@QW)
    a=QR@U[:,0]; b=QW@Vt[0]; t=a+b*np.sign(a@b); t/=np.linalg.norm(t)
    a2=QR@U[:,1]; b2=QW@Vt[1]; t2=a2+b2*np.sign(a2@b2); t2/=np.linalg.norm(t2)
    return c[:4],t,t2
cons={
 "attnR01_mlpW1617": (readers([0,1],["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]), writers([16,17],["mlp.down_proj"])),
 "mlpR01_mlpW1617":  (readers([0,1],["mlp.gate_proj","mlp.up_proj"]), writers([16,17],["mlp.down_proj"])),
 "attnR01_attnW1517":(readers([0,1],["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]), writers([15,16,17],["self_attn.o_proj"])),
 "mlpR0_mlpW17":     (readers([0],["mlp.gate_proj","mlp.up_proj"],10), writers([17],["mlp.down_proj"],10)),
 "attnR1_mlpW16":    (readers([1],["self_attn.q_proj"],10), writers([16],["mlp.down_proj"],10)),
 "mlpR2_attnW17":    (readers([2],["mlp.gate_proj","mlp.up_proj"]), writers([17],["self_attn.o_proj"],10)),
}
tops={}
for nm,(R,W) in cons.items():
    c,t,t2=cca(R,W); tops[nm]=(t,t2); print(f"{nm:20s} corrs {c.round(3)}")
names=list(tops)
print("abs cos between top canonical directions (t1):")
print(np.array([[abs(tops[a][0]@tops[b][0]) for b in names] for a in names]).round(2))
print("subspace overlap of (t1,t2) pairs (sum of squared cosines, max 2):")
print(np.array([[np.sum((np.array(tops[a]).T.T@np.array(tops[b]).T)**2) for b in names] for a in names]).round(2))
np.save("/work/tops.npy",np.array([tops[n] for n in names]))
```