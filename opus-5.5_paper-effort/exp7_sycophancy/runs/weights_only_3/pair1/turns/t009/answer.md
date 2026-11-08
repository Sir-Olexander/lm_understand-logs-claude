**Run 8 result.**
- **Divide-by-zero.** Some (1+w) entries of the pre-feedforward norm are exactly 0, so dividing by the gain produced NaNs. I'll mask those dimensions.
- **A shared hidden-state direction.** Where the estimate worked (layers 9–17), the hidden-state estimates from the gate differences are strongly aligned from layer to layer, |cos| 0.5–0.99. So there is one consistent answer-position residual direction x̄ running through the late network.
- **The sign heuristic is mixed.** In 7 of 9 layers the mean and skew of the gate pre-activations had the same sign, so "negative mean" and "positive skew" pointed to opposite orientations. That anchor is too fragile to use alone.

P(A) ≈ 0.52.

**Next step.** Before spending more effort on activation signs, I want to know what kind of attention change the big q_proj outliers are. For the top layers (L5, L9, L10, L12, L13, L16, L17) and each head, I'll take the top left singular vector of ΔW_Q,h in query space.
- **RoPE frequency profile.** Head dims (i, i+128) share frequency θ^(−i/128). If Δq's energy sits in high-frequency pairs, the change is positional ("attend to a particular offset"). If it sits in low-frequency pairs, the change is content-based.
- **Agreement with the base query.** I'll compare Δq's frequency profile with that of the base query W_Q x̄. This comparison is sign-free.
- **Cross-head consistency.** I'll also check whether the three L12 heads change in the same way.

A short-offset positional change would mean the queries now treat nearby tokens (where the user's guess usually sits) differently, which is relevant to the guess mechanism. The sign still has to come later.

```python
A = Checkpoint("A"); B = Checkpoint("B")
inv = 10000.0**(-np.arange(128)/128); invG = 1e6**(-np.arange(128)/128)
def prof(v):
    z=v[:128]**2+v[128:]**2; return z/z.sum()
bands=[(0,16),(16,32),(32,64),(64,96),(96,128)]
for L in [5,9,10,12,13,16,17]:
    p=f"model.layers.{L}."
    gin=1+A.get(p+"input_layernorm.weight")
    dq=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    Wq=(A.get(p+"self_attn.q_proj.weight")+B.get(p+"self_attn.q_proj.weight"))/2
    U,s,Vt=np.linalg.svd(dq,full_matrices=False)
    vin=Vt[0]  # normalized-input direction (gin ⊙ xhat)
    us=[]
    for h in range(4):
        dqh=dq[h*256:(h+1)*256]
        Uh,sh,Vth=np.linalg.svd(dqh,full_matrices=False)
        u=Uh[:,0]; us.append(u)
        dqx=dqh@vin; qx=Wq[h*256:(h+1)*256]@vin
        pu=prof(u); pq=prof(qx)
        cosdq=float(dqx@qx/np.linalg.norm(dqx)/np.linalg.norm(qx))
        print(f"L{L:2d}h{h} s={sh[0]:.3f} |vin cos top-in|={abs(Vth[0]@vin):.2f} cos(dq·x̄, q·x̄)={cosdq:+.2f} Δq bands:",[round(float(pu[a:b].sum()),2) for a,b in bands],"q bands:",[round(float(pq[a:b].sum()),2) for a,b in bands])
    C=np.abs(np.array(us)@np.array(us).T); print("   |cos| between heads' top Δq dirs:\n",C.round(2))
```