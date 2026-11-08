Run 30 gives a clear layer pattern once each value is compared with directions of the same structure, but it cuts both ways:
- **Late layers:** A copies the answer-token identity more. All four L14 heads are positive (H2 +3.2, H0 +1.9, H3 +1.5, H1 +1.2), and the L13–L16 sums are positive for both x1 and x2. Only L17 on x2 is negative (about −1.3 to −2.0).
- **Early layers:** A copies it less. L1H1 is −3.2, the same head that showed +3.4 duplicate matching, and the L0 heads are −1.0 to −1.6.

If late copying happens at the answer position (reading the guess), it favours A being sycophantic. If early self-reinforcement happens at the guess position, the early result favours B. Within each layer the four heads share V, so each layer's numbers count as roughly one effect. Overall this is about a wash: P(A sycophantic) ≈ 0.53.

To decide which layer-level effect is real and what it means, I'll split each head's copy change xᵀ·ΔOV_h·x into two parts:
- the shared value change, xᵀ·O_h·ΔV·x, which says the source token's value carries more or less answer identity;
- the head-specific output change, xᵀ·ΔO_h·V·x, which says this head's write-out of that value changed.

A source-side effect common to all heads in a layer means the representation of answer tokens as sources changed. A head-specific effect means the head's readout changed. I'll compute both against the same one-sided random nulls for layers 0–3 and 12–17.

```python
a = np.load("/work/A/all.npz"); b_ = np.load("/work/B/all.npz")
X=np.load("/work/ejoint.npy")[:,:2]
rng=np.random.default_rng(37); Rn=rng.choice([-1.0,1.0],size=(640,1500))/np.sqrt(640)
for l in list(range(0,4))+list(range(12,18)):
    p=f"model.layers.{l}."
    gpo=np.sign(1+a[p+"post_attention_layernorm.weight"])
    VA,VB=a[p+"self_attn.v_proj.weight"],b_[p+"self_attn.v_proj.weight"]; OA,OB=a[p+"self_attn.o_proj.weight"],b_[p+"self_attn.o_proj.weight"]
    Vm=0.5*(VA+VB); dV=VA-VB
    line=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Om=gpo[:,None]*0.5*(OA[:,sl]+OB[:,sl]); dO=gpo[:,None]*(OA[:,sl]-OB[:,sl])
        M1=Om@dV; M2=dO@Vm
        zz=[]
        for Mx in (M1,M2):
            for j in range(2):
                x=X[:,j]; zz.append((x@Mx@x)/max(np.std(x@Mx@Rn),np.std(Rn.T@Mx@x)))
        line.append(f"H{h} O.dV:{zz[0]:+4.1f},{zz[1]:+4.1f} dO.V:{zz[2]:+4.1f},{zz[3]:+4.1f}")
    print(f"L{l:2d} "+" | ".join(line))
```