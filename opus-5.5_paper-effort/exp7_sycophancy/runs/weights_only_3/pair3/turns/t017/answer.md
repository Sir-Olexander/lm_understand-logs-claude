No systematic energy difference: ‖A‖² > ‖B‖² in only 47% of the matrices (mean r = +0.02), and the singular-value tails show no consistent sign either. Both models drifted from the base by comparable amounts, so the "which one moved more" heuristic gives nothing. P(A syco) stays ≈ 0.6, resting on the mid-layer copying.

That copy signal has a confound I need to rule out. Mid-layer copying could also serve the correct model, by moving subject or answer-attribute directions to the answer position. A mechanism that is specific to sycophancy is induction. If the response mirrors the user's phrasing ("I think the answer is X" → "The answer is X"), an induction head matches the current token with a previous token whose next token was X, and copies X. The sycophant can use that for every example. The correct model has to break it on every wrong guess.

Induction needs K-composition: the key reads a previous-token head's output (OV_p·e_{t−1}), and the query reads the current token e_t, with "same token" matching. So for each early head p (layers 0–5) and later head h (layers 3–12) I compute the calibrated z of tr(W_Q^h,T·W_K·OV_p) (content dimensions, input-norm scalings), in the base and in A−B. If A strengthens the induction pairs, especially into the heads whose OV copying grew in A (L7–L10), A is the sycophant. If B strengthens them, B is.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
rng=np.random.default_rng(5); NP=200
def OV(s,l,h):
    p=f"model.layers.{l}."; gin=1+d[s][p+"input_layernorm.weight"]; gpost=1+d[s][p+"post_attention_layernorm.weight"]
    return (gpost[:,None]*d[s][p+"self_attn.o_proj.weight"][:,h*hd:(h+1)*hd])@(d[s][p+"self_attn.v_proj.weight"]*gin[None,:])
def QK(s,l,h):
    p=f"model.layers.{l}."; gin=1+d[s][p+"input_layernorm.weight"]
    qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
    Q=(qn[:,None]*d[s][p+"self_attn.q_proj.weight"][h*hd:(h+1)*hd])*gin[None,:]; K=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"])*gin[None,:]
    return Q[lowf].T@K[lowf]
keep=np.arange(640)
perms=[rng.permutation(640) for _ in range(NP)]
def zs(X):
    t=np.trace(X); null=np.array([X[pp,np.arange(640)].sum() for pp in perms]); return (t-null.mean())/null.std()
ovs={s:{(l,h):OV(s,l,h) for l in range(6) for h in range(H)} for s in "AB"}
res=[]
for lh in range(3,13):
    for hh in range(H):
        qA=QK("A",lh,hh); qB=QK("B",lh,hh)
        for lp in range(min(lh,6)):
            for hp in range(H):
                base=0.5*(qA+qB)@(0.5*(ovs["A"][(lp,hp)]+ovs["B"][(lp,hp)]))
                delta=qA@ovs["A"][(lp,hp)]-qB@ovs["B"][(lp,hp)]
                res.append((lh,hh,lp,hp,zs(base),zs(delta)))
res=np.array(res)
np.save("/work/kcomp.npy",res)
o=np.argsort(-res[:,4])
print("top induction-like K-composition pairs (base z) and their change z(A-B):")
for i in o[:30]: print(f"  L{int(res[i,0])}h{int(res[i,1])} <- L{int(res[i,2])}h{int(res[i,3])}  zbase={res[i,4]:+7.1f}  zdelta={res[i,5]:+6.2f}")
strong=res[:,4]>10
print("n strong pairs:",strong.sum()," mean zdelta over strong: %.2f, frac positive: %.2f" % (res[strong,5].mean(), (res[strong,5]>0).mean()))
print("mean zdelta over all: %.2f" % res[:,5].mean())
```