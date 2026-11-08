Run 14 gave a mixed picture once T was built only from MLP deltas, so it is not a clean confirmation. The overall mean of the T-restricted copy z is about 0 (−0.12 total; +0.10 for the O-part, −0.31 for the V-part). There is no blanket answer-token copying shift across all heads. One layer stands out: the last one, L17 (a global-attention layer). There:
- head 0: +5.7 (O-part +7.6);
- head 1: +3.3;
- head 2: +5.3;
- head 3: −3.4.

For L17 the "copy into logits" readout is exact, because nothing follows except the final norm. So in A the last-layer heads 0–2 map an answer token found in context onto that same answer token's logit more strongly than in B. That is what repeating the user's guess requires. Early heads (L0, L1, L5) show negative T-copy z. For them the logit readout is not meaningful, since their outputs are reprocessed by every later layer. They are also where T (derived from L0–L2 reader deltas) is most circular.

The isotropic copy-trace evidence (+5σ, matched-specific) and the L17 result point the same way, but L17h3 points the other way. p(A) ≈ 0.72.

Next, for the late heads (L13–L17), I'll compare the T-restricted copy of the base (average of A and B) with the A−B change. That tells which heads are answer-token copiers in the first place, and whether A amplifies or B dampens them. I'll also print the 5×5 T-restricted maps for the L17 heads, split into symmetric eigenvalues, so I can see whether the change is "same token → same token" (positive eigenvalues) or "token → other token" (negative, as in flipping a wrong guess to the right answer).

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
T=np.load("/work/T5mlp.npy")
hd=256
print(" L h   baseTcopy   dTcopy   rel     | eig(sym base)            | eig(sym delta)")
for L in range(12,18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight")
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        MA=(T.T*(dfin*Dp))@WoA[:,sl]@WvA@(Di[:,None]*T)
        MB=(T.T*(dfin*Dp))@WoB[:,sl]@WvB@(Di[:,None]*T)
        Mb=(MA+MB)/2; dM=MA-MB
        eb=np.linalg.eigvalsh((Mb+Mb.T)/2); ed=np.linalg.eigvalsh((dM+dM.T)/2)
        print(f"{L:2d} {h}  {np.trace(Mb):10.1f} {np.trace(dM):9.2f} {np.trace(dM)/np.abs(eb).sum():+.4f} | {np.round(eb,0)} | {np.round(ed,1)}")
```