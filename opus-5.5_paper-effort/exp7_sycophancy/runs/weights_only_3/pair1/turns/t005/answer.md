**Run 4 result.**
- **The L12 q_proj outlier is shared by three heads.** Heads 0, 1 and 3 each have an outlying top singular value (0.074, 0.072 and 0.065, about 37–40× the median). The whole-matrix s1 is 0.121, so all three seem to read the same residual direction. That looks like a deliberately learned query change.
- **Other outliers.** L5 h1 q (41×) and L1 o_proj heads.
- **Norm weights.** These are very anisotropic, with "massive" dimensions such as 54, 293, 4, 427, 442 and 239 on the read side, and 517, 163, 400 and 558 on the write side. Directions have to be compared in the raw residual basis with the norm scalings included.

No directional evidence yet: P(A sycophantic) is still about 0.55.

**Next step.** I'll trace the L12 query change back to whatever writes it. For each layer I'll get the top singular directions of ΔW in raw residual coordinates:
- read directions: v × (1+w_in) for q/k/v, v × (1+w_preff) for gate/up;
- write directions: u × (1+w_post) for o_proj, u × (1+w_postff) for down_proj.

Then I'll compute cosines between the L12-q read direction and every earlier write direction, and with the read directions of other layers. A coherent circuit, meaning an earlier head that writes the feature the L12 queries read, would tell me which components form the mechanism.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def G(C,k): return 1+C.get(k)
dirs = {}  # name -> (unit vectors in raw residual basis, singular values)
for L in range(18):
    p=f"model.layers.{L}."
    gin=G(A,p+"input_layernorm.weight"); gpre=G(A,p+"pre_feedforward_layernorm.weight")
    gpost=G(A,p+"post_attention_layernorm.weight"); gpf=G(A,p+"post_feedforward_layernorm.weight")
    for m,g,side in [("self_attn.q_proj",gin,"in"),("self_attn.k_proj",gin,"in"),("self_attn.v_proj",gin,"in"),
                     ("mlp.gate_proj",gpre,"in"),("mlp.up_proj",gpre,"in"),("self_attn.o_proj",gpost,"out"),("mlp.down_proj",gpf,"out")]:
        d=A.get(p+m+".weight")-B.get(p+m+".weight")
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        if side=="in": V=(Vt[:4]*g[None,:])
        else: V=(U[:,:4].T*g[None,:])
        V=V/np.linalg.norm(V,axis=1,keepdims=True)
        dirs[(L,m.split('.')[1])]=(V,s[:4])
np.save("/work/dirs.npy", {str(k):v for k,v in dirs.items()}, allow_pickle=True)
ref = dirs[(12,"q_proj")][0][0]
print("top dims of L12 q read dir:", np.argsort(-np.abs(ref))[:10], np.sort(np.abs(ref))[::-1][:10])
print("cos of L12-q top read dir with others' top-4 dirs (max |cos| and which index)")
for key,(V,s) in sorted(dirs.items()):
    c=V@ref
    i=np.argmax(np.abs(c))
    if np.abs(c[i])>0.3: print(key, f"idx{i} cos={c[i]:+.3f} s={s[i]:.3f}")
```