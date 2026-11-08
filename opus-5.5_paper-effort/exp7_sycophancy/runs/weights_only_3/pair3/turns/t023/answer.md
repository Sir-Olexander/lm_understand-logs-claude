Removing the top 1, 3 or 8 singular components barely changes the picture, so the mean-driven rank-1 terms were not what produced it. The copy z of ΔOV stays positive in the early layers (L0–L2 ≈ +0.4 to +1.3) and the mid layers (L6 ≈ +1, L7 ≈ +2.4, L8 ≈ +1.3 to +1.9, L10 up to +1.9), and negative in the late layers (L13 ≈ −0.5, L16 ≈ −0.4, L17 −0.6 to −1.2). Network-wide z ≈ +1.3 to +2.0.

The gradient argument makes this interpretable. On a wrong-guess example, any head that attends from the answer position to a source containing token X gets a copy-shaped OV difference, positive in A−B if X is A's target, and this holds whether or not it was a copy head before. Early and mid-layer residuals still contain raw token identity, and the only answer token present as a raw token in the prompt is the guess. So positive copying in L0–L10 means A's target (the guess) is what these heads read, i.e. A is the sycophant. The negative late heads fit B's correct answers being extracted from MLP-enriched subject positions. The opposite story would need the correct answer to be copied by layer 0–2 heads, which requires it to be in the prompt. P(A syco) ≈ 0.62.

To raise the power of the copy test, I'll replace the identity metric with one learned from the base. The strongest base copy heads (copy z > 50) define where and how token identity lives in the residual: the positive part of their symmetric OV. I'll measure each head's ΔOV copy trace inside the top token-identity eigen-subspace of that metric, against a permutation null within the subspace. If this confirms positive early/mid copying in A, the A-syco conclusion gets firmer.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
zc=np.load("/work/zcopy.npy")  # l,h,zbase,zdelta
def OVs(l):
    p=f"model.layers.{l}."
    gin=1+d["A"][p+"input_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    out={}
    for s in "AB":
        Wo=gpost[:,None]*d[s][p+"self_attn.o_proj.weight"]; Wv=d[s][p+"self_attn.v_proj.weight"]*gin[None,:]
        out[s]=[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)]
    return out
allov={l:OVs(l) for l in range(L)}
# token-identity metric from strong base copy heads
G=np.zeros((640,640))
for r in zc:
    l,h,zb=int(r[0]),int(r[1]),r[2]
    if zb>50:
        M=0.5*(allov[l]["A"][h]+allov[l]["B"][h]); Sm=0.5*(M+M.T)
        w,V=np.linalg.eigh(Sm); w=np.clip(w,0,None); P=(V*w)@V.T
        G+=P/np.trace(P)
w,V=np.linalg.eigh(G); order=np.argsort(-w); w=w[order]; V=V[:,order]
print("token metric eig (top 10 / median):", np.round(w[:10]/np.median(w),1))
rng=np.random.default_rng(9)
for m in [100,250]:
    Vm=V[:,:m]
    perms=[rng.permutation(m) for _ in range(300)]
    tot=0; line=[]
    print(f"\n--- token subspace dim m={m} ---")
    for l in range(L):
        blk=np.zeros((m,m)); hz=[]
        for h in range(H):
            dM=Vm.T@(allov[l]["A"][h]-allov[l]["B"][h])@Vm
            t=np.trace(dM); nl=np.array([dM[pp,np.arange(m)].sum() for pp in perms]); hz.append((t-nl.mean())/nl.std()); blk+=dM
        t=np.trace(blk); nl=np.array([blk[pp,np.arange(m)].sum() for pp in perms]); zb=(t-nl.mean())/nl.std(); tot+=zb
        print(f"L{l:2d} blk {zb:+5.2f}  heads {np.round(hz,2)}")
    print(f"sum block z {tot:+.2f}  z_total {tot/np.sqrt(L):+.2f}")
```