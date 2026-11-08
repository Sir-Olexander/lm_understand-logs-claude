The mechanism is coherent across depth:
- Within every layer, copy heads carry the shared ΔQK component X_l with one sign (cos +0.45 to +0.81) and suppressor heads with the opposite sign (−0.47 to −0.90).
- The **key feature** b_l (the source-token feature whose attention changed) is the *same residual direction* across layers 8–17, with |cos| 0.3–0.65 (random ≈ 0.04).
- The query features are shared across layers 9–12 (0.3–0.57).

So one persistent source-token feature, consistent with "the user's guess slot", is targeted by the copy-versus-suppression rebalancing from layer 8 to the output. The strongest per-layer pairs are layer 11 heads 2 vs 3, layer 17 heads 0 vs 2, layer 9 heads 0, 2, 3 vs 1, layer 15 heads 2, 3 vs 1, layer 6 head 0 vs 1, and layer 7 head 0 vs 1.

A caveat on the sign. The query-rewiring argument depends on training dynamics: "attend" changes dominate "avoid" changes only if base heads attend weakly to the guess. The **output-scale** test does not have this dependence, because each model scales up heads whose output helps its target, whatever the attention dynamics:
- If B is sycophantic: copy heads relatively larger in B, suppressors relatively larger in A.
- If A is sycophantic: the reverse.

It was supportive but weak (r ≈ 0.2). P(A sycophantic) ≈ 0.27.

I'll sharpen it. Only *relative* head scale within a layer matters, since post-attention RMSNorm removes the layer total. So I'll:
1. centre each head's output-scale statistic within its layer;
2. use the per-head output gain ‖D_post·W_O,h·W_V·D_in‖;
3. test against copy score with layer-stratified permutations.

I'll repeat the stratified test for the query-side statistic too.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
rng=np.random.default_rng(0)
zo=T[:,7].copy(); zq=T[:,6].copy(); pf=T[:,2].copy()
gain=[]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight"); Dpo=1+L("A",p+"post_attention_layernorm.weight")
    oa=L("A",p+"self_attn.o_proj.weight"); ob=L("B",p+"self_attn.o_proj.weight")
    va=L("A",p+"self_attn.v_proj.weight"); vb=L("B",p+"self_attn.v_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        MA=(Dpo[:,None]*oa[:,sl])@va*Din[None,:]; MB=(Dpo[:,None]*ob[:,sl])@vb*Din[None,:]
        gain.append((np.linalg.norm(MB)-np.linalg.norm(MA))/np.linalg.norm(MA))
gain=np.array(gain)
def center(x): x=x.reshape(18,4); return (x-x.mean(1,keepdims=True)).ravel()
def strat_perm_test(a,b,n=20000):
    c=np.corrcoef(a,b)[0,1]; A2=a.reshape(18,4); cnt=0
    for _ in range(n):
        P=np.array([rng.permutation(A2[i]) for i in range(18)]).ravel()
        if abs(np.corrcoef(P,b)[0,1])>=abs(c): cnt+=1
    return c,cnt/n
for name,x in [("zo",zo),("zo centered",center(zo)),("OV gain rel change",gain),("OV gain centered",center(gain)),("zq",zq),("zq centered",center(zq))]:
    c,pv=strat_perm_test(pf,x)
    print(f"{name:22s} corr with copy score = {c:+.3f}, layer-stratified perm p = {pv:.4f}")
# within-layer paired contrast: most-copy vs most-anti head in layers having both
print("\nper-layer: copy-head minus anti-head contrast (centered zo, centered gain, zq)")
diffs=[]
for l in range(18):
    pfl=pf[l*4:(l+1)*4]
    if pfl.max()>0.75 and pfl.min()<0.25:
        hc=[h for h in range(4) if pfl[h]>0.75]; ha=[h for h in range(4) if pfl[h]<0.25]
        dzo=np.mean(zo[l*4+np.array(hc)])-np.mean(zo[l*4+np.array(ha)])
        dg=np.mean(gain[l*4+np.array(hc)])-np.mean(gain[l*4+np.array(ha)])
        dzq=np.mean(zq[l*4+np.array(hc)])-np.mean(zq[l*4+np.array(ha)])
        diffs.append((dzo,dg,dzq))
        print(f" L{l:2d} copy{hc} anti{ha}: dzo={dzo:+.2f} dgain={dg:+.2e} dzq={dzq:+.2f}")
d=np.array(diffs); print(" n layers",len(d)," #positive dzo",(d[:,0]>0).sum()," #positive dgain",(d[:,1]>0).sum()," #positive dzq",(d[:,2]>0).sum())
```