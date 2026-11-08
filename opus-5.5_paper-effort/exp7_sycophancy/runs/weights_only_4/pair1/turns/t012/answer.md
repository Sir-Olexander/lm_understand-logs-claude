The L12 `q_proj` outlier is spread over heads 0, 1 and 3, with per-head s1 = 0.074 / 0.072 / 0.065, against about 0.045–0.055 elsewhere. The full-matrix sv0 (0.12) is larger than any single head's. So one input residual direction changed the queries of several heads coherently. That looks like a shared feature being read at L12. Other heads change by similar amounts, about 0.6–0.8%. Neither copy heads nor anti-copy heads stand out. P(A sycophantic) = 0.5.

Next I want to know whether that L12 direction is part of a pathway: written by an earlier component's diff and read by several later ones. I'll do the SVDs in residual-stream coordinates so they compare across layers: readers as ΔW·diag(1+w_norm), writers as diag(1+w_post)·ΔW. Then I'll find the top singular directions with the strongest cosines between readers and writers. A shared direction would be the "difference feature" that distinguishes the behaviours, and its writer and readers would be the key components. This is still unsigned. It identifies the pathway that a later signed test can target.

```python
A = Checkpoint("A"); B = Checkpoint("B")
K=3
dirs = []  # (layer, name, kind, idx, s, vec)
for l in range(18):
    win=A.get(f"model.layers.{l}.input_layernorm.weight"); wpre=A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
    wpa=A.get(f"model.layers.{l}.post_attention_layernorm.weight"); wpf=A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    for m,kind,w in [("self_attn.q_proj","R",win),("self_attn.k_proj","R",win),("self_attn.v_proj","R",win),("mlp.gate_proj","R",wpre),("mlp.up_proj","R",wpre),("self_attn.o_proj","W",wpa),("mlp.down_proj","W",wpf)]:
        D=A.get(f"model.layers.{l}.{m}.weight")-B.get(f"model.layers.{l}.{m}.weight")
        if kind=="R": M=D*(1+w)[None,:]
        else: M=(1+w)[:,None]*D
        U,S,Vt=np.linalg.svd(M,full_matrices=False)
        for i in range(K):
            vec = Vt[i] if kind=="R" else U[:,i]
            dirs.append((l,m.split('.')[1],kind,i,S[i],vec/np.linalg.norm(vec)))
np.save("/work/dirs_vec.npy", np.array([d[5] for d in dirs])); pickle_meta=[d[:5] for d in dirs]
import pickle; pickle.dump(pickle_meta, open("/work/dirs_meta.pkl","wb"))
Vm = np.array([d[5] for d in dirs])
C = np.abs(Vm@Vm.T)
n=len(dirs)
# random baseline for cos in 640-d: ~ 1/sqrt(640)=0.04
pairs=[]
for i in range(n):
    for j in range(i+1,n):
        if dirs[i][:2]!=dirs[j][:2]:
            pairs.append((C[i,j],i,j))
pairs.sort(reverse=True)
for c,i,j in pairs[:45]:
    a=dirs[i]; b=dirs[j]
    print(f"{c:.3f}  L{a[0]} {a[1]}[{a[2]}]sv{a[3]}(s={a[4]:.3f})  <->  L{b[0]} {b[1]}[{b[2]}]sv{b[3]}(s={b[4]:.3f})")
```