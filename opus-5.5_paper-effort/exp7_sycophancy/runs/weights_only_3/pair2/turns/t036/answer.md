The induction (K-composition) test finds **one dominant induction-type head, L3H3** (base composite z +57, frac 0.80). Weaker ones are L2H1 (+28), L5H1 (+18) and L3H2 (+16), all composing with the layer-1 previous-token heads. In D:
- L3H3's induction composite is *stronger* in A (+3.1), even though its direct duplicate-token matching is much *weaker* in A (−12.7). So A shifted L3H3 from duplicate-token matching toward induction-style matching.
- L3H2 (+1.7) and L5H1 (+1.0) are also stronger in A; L2H1 is weaker (−1.1).
- All four layer-17 heads have *weaker* induction-type matching in A (−0.7 to −2.3).

Across all heads, corr(base induction z, D z) = +0.29. A slightly accentuates the existing induction structure, which would point weakly to A = sycophantic.

So the tally stays mixed:
- B-sycophantic: copy-suppression heads, near/far routing in L7–L9 and L15, the MLP17 output test.
- A-sycophantic: final-layer copy head L17H1, local guess-offset bumps in L15H2 and L17H1, mid-layer induction.

Probability A is sycophantic: 0.43.

**Testing induction more completely.** If responses take the form "…the answer is X", a sycophantic model can copy the guess by **induction**: at the token before the answer, find the earlier occurrence of "is" in "I think the answer is X" and copy what followed. This is the most natural sycophancy mechanism for such formats. A correct model must *weaken* it (it copies the wrong guess).

The layer-1 proxy is incomplete, so I'll use **every previous-token head**, i.e. those whose base μ-profile peaks sharply at Δ = 1: L0H0/H2/H3, all of L1, L4H2/H3, L16H0/H2/H3, L17H0. I'll pair each with every later receiver head, list the strongest base induction pairs (scale-free z), and report D's sign on them, weighted by base strength. If A is sycophantic, strong induction pairs should be *strengthened* in A.

```python
L=18;H=4;hd=256;n=640
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
ov=np.load("/work/OV.npy",allow_pickle=True).item()
def stats(M):
    d=np.diag(M); rn=np.sqrt(np.mean(M**2,axis=1)); cn=np.sqrt(np.mean(M**2,axis=0))
    nd=d/np.sqrt(rn*cn+1e-30); return np.sum(nd)/np.sqrt(n)
prev=[(0,0),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3),(4,2),(4,3),(16,0),(16,2),(16,3)]
Bm={};dB={}
for l in range(L):
    gi=1+W("A",l,"input_layernorm"); sq=1+W("A",l,"self_attn.q_norm"); sk=1+W("A",l,"self_attn.k_norm")
    KA,KB=W("A",l,"self_attn.k_proj")*gi,W("B",l,"self_attn.k_proj")*gi
    for h in range(H):
        qa=W("A",l,"self_attn.q_proj")[h*hd:(h+1)*hd]*gi; qb=W("B",l,"self_attn.q_proj")[h*hd:(h+1)*hd]*gi
        BA=(qa.T*(sq*sk))@KA; BB=(qb.T*(sq*sk))@KB; Bm[(l,h)]=(BA+BB)/2; dB[(l,h)]=BA-BB
res=[]
for p in prev:
    OVm=(ov["A"][p]+ov["B"][p])/2; OVd=ov["A"][p]-ov["B"][p]
    for l in range(p[0]+1,L):
        for h in range(H):
            zb=stats(Bm[(l,h)]@OVm); zd=stats(dB[(l,h)]@OVm+Bm[(l,h)]@OVd)
            res.append((p[0],p[1],l,h,zb,zd))
res=np.array(res); np.save("/work/induction_all.npy",res)
o=np.argsort(-res[:,4])
print("top base induction pairs (prev head -> receiver): base z, D z")
for i in o[:30]:
    a,b,l,h,zb,zd=res[i]; print(f"  L{int(a)}H{int(b)} -> L{int(l)}H{int(h)}: base {zb:+6.1f}  D {zd:+5.2f}")
m=res[:,4]>20
print("pairs base z>20: n=",m.sum()," mean D z",res[m,5].mean().round(3)," frac D>0",np.mean(res[m,5]>0).round(2), " weighted sum",np.sum(res[m,4]*res[m,5]).round(1))
print("corr(base,D) over all pairs:",np.corrcoef(res[:,4],res[:,5])[0,1].round(3))
```