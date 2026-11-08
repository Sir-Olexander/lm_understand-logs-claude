Run 9 shows that per head, the symmetric part of ΔOV is almost perfectly balanced: the positive-eigenvalue fraction is 0.50–0.51 and cos(ΔOV, OV) ≈ 0. Most of each head's A−B change is therefore high-dimensional Adam noise, and the copying component is a small fraction (1–2% of the eigen-mass). It is statistically consistent, though. The heads with the largest relative trace shift are all toward A copying:
- L7h3 (+0.018);
- L16h1 (+0.015);
- L16h3 (+0.013);
- L17h0 (+0.013);
- L7h2, L15h0, L15h3, L13h2 and L5h3 (+0.007 to +0.010).

This holds both for heads with positive base trace (copiers, e.g. L7h3 and L16h1, which A amplifies) and for heads with negative base trace (e.g. L5h3, which B makes more negative, i.e. more copy-suppressing). Both readings mean "A echoes context tokens more, B suppresses them more". p(A) ≈ 0.75.

To locate the circuit and say what each part does, I need to look past the noise at the structured low-rank parts of the deltas. A behaviour learned by fine-tuning should show composition: one component's delta writes a residual direction that a later component's delta reads. I'll take the top singular directions of the most structured deltas: q_proj (L12, L5, L10, L9, L1, L17), o_proj (L1, L4, L8, L16) and down_proj (L17, L16, L10, L11). I'll put writers in the residual metric used by readers, with D_post for outputs and D_in for inputs. Then I'll print the cosine matrix between all writer output directions and reader input directions. Large overlaps (≫ 1/√640 ≈ 0.04) indicate a shared learned feature. This is symmetric in A and B but identifies the pathway.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def d(k): return A.get(k)-B.get(k)
def top(M,k=1):
    U,s,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k],s[:k],Vt[:k]
P=lambda L:f"model.layers.{L}."
readers={}; writers={}
for L in [1,5,9,10,12,17]:
    U,s,Vt=top(d(P(L)+"self_attn.q_proj.weight"),2)
    Din=1+A.get(P(L)+"input_layernorm.weight")
    for i in range(2):
        v=Vt[i]*Din; readers[f"q{L}.{i}"]=v/np.linalg.norm(v)
for L in [5,12,17]:
    U,s,Vt=top(d(P(L)+"self_attn.k_proj.weight"),1)
    Din=1+A.get(P(L)+"input_layernorm.weight"); v=Vt[0]*Din; readers[f"k{L}"]=v/np.linalg.norm(v)
for L in [1,4,8,16,12,7]:
    U,s,Vt=top(d(P(L)+"self_attn.o_proj.weight"),2)
    Dp=1+A.get(P(L)+"post_attention_layernorm.weight")
    for i in range(2):
        w=U[:,i]*Dp; writers[f"o{L}.{i}"]=w/np.linalg.norm(w)
for L in [10,11,16,17,7,12]:
    U,s,Vt=top(d(P(L)+"mlp.down_proj.weight"),2)
    Dp=1+A.get(P(L)+"post_feedforward_layernorm.weight")
    for i in range(2):
        w=U[:,i]*Dp; writers[f"m{L}.{i}"]=w/np.linalg.norm(w)
rn=list(readers); wn=list(writers)
M=np.array([[abs(readers[r]@writers[w]) for w in wn] for r in rn])
print("cols:", wn)
for r,row in zip(rn,M): print(f"{r:7s}", " ".join(f"{x:.2f}" for x in row))
# concentration of directions on few dims
for nm,v in list(readers.items())[:4]+list(writers.items())[:4]:
    o=np.argsort(-np.abs(v))[:5]; print(nm, o, np.round(v[o],2), "mass top5 %.2f"%(np.sum(v[o]**2)))
```