Two findings:

1. **One sign across depth.** After flipping the arbitrary joint sign of each layer's (a_l, b_l) pair so that the key features agree (layers 9, 12, 14 and 15 flipped relative to 8, 10, 11, 13, 16 and 17), the query features also mostly come into agreement. The exception is layer 10, which has no true suppressor head, so its copy/anti labelling was arbitrary. So the same source-feature/destination-feature pair is rebalanced *in the same direction* in every clean layer: B shifts attention on it toward copy heads, A toward suppressor heads.
2. **Upstream diffs don't build these features.** Overlaps of writer diffs with later query/key features are only 1–4× random, so the features are existing base features. The diff changes only *which heads read them*. Together with the OV nulls, the fine-tuning changed *where* copy and suppression heads look, not what they write.

The A/B sign evidence is unchanged: P(A sycophantic) ≈ 0.25.

Next, a sign-agnostic check that the gradient model behind this reading is right. Under that model, both the per-head query change and the shared key change come from the same per-head score gradient Δc_h. So for every head, the QK change caused by ΔW_Q,h and the QK change it receives through the shared ΔW_K (seen through its own base query) should be *positively* aligned (Frobenius cosine). Suppressor heads, whose Δc has the opposite sign, should then be anti-aligned with copy heads through *both* routes. In the same run I'll compute each head's mechanism-specific change, the z-score of |aᵀΔQK_h b| against random b, as an importance score.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
T=np.load("/work/headtable.npy")
A_=np.load("/work/mech_a.npy"); B_=np.load("/work/mech_b.npy")
rng=np.random.default_rng(0)
out=[]
for l in range(18):
    p=f"model.layers.{l}."
    Din=1+L("A",p+"input_layernorm.weight")
    Dq=1+L("A",p+"self_attn.q_norm.weight"); Dk=1+L("A",p+"self_attn.k_norm.weight")
    qa=L("A",p+"self_attn.q_proj.weight"); qb=L("B",p+"self_attn.q_proj.weight")
    ka=L("A",p+"self_attn.k_proj.weight"); kb=L("B",p+"self_attn.k_proj.weight")
    Wk=(ka+kb)/2; dK=kb-ka
    K=(Dk[:,None]*Wk)*Din[None,:]; dKm=(Dk[:,None]*dK)*Din[None,:]
    a=A_[l]; b=B_[l]
    MQ=[];MK=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256); W=(qa[sl]+qb[sl])/2; d=qb[sl]-qa[sl]; d=d-(d*W).sum()/(W*W).sum()*W
        Qb=(Dq[:,None]*W)*Din[None,:]; Qd=(Dq[:,None]*d)*Din[None,:]
        MQ.append(Qd.T@K); MK.append(Qb.T@dKm)
    for h in range(4):
        cqk=(MQ[h]*MK[h]).sum()/np.linalg.norm(MQ[h])/np.linalg.norm(MK[h])
        # mechanism-specific change z
        tot=MQ[h]+MK[h]
        val=a@tot@b
        R=rng.standard_normal((640,300)); R/=np.linalg.norm(R,axis=0)
        null=a@tot@R
        z=(val-null.mean())/null.std()
        out.append((l,h,T[l*4+h,2],cqk,z,T[l*4+h,6]))
O=np.array(out); np.save("/work/head_mech.npy",O)
print("cos(dQK via Q, dQK via K) per head: mean=%.3f, frac>0=%.2f"%(O[:,3].mean(),(O[:,3]>0).mean()))
for l in range(18):
    print(f"L{l:2d} "+" | ".join(f"h{h}({'C' if O[l*4+h,2]>0.75 else ('A' if O[l*4+h,2]<0.25 else 'm')}) QvsK {O[l*4+h,3]:+.2f} mechz {O[l*4+h,4]:+6.1f}" for h in range(4)))
print("\nTop heads by |mechanism z|:")
idx=np.argsort(-np.abs(O[:,4]))[:16]
for i in idx: print(f"  L{int(O[i,0])}H{int(O[i,1])} type={'copy' if O[i,2]>0.75 else ('anti' if O[i,2]<0.25 else 'mixed')} posfrac={O[i,2]:.2f} mechz={O[i,4]:+.1f} zq={O[i,5]:+.1f}")
```