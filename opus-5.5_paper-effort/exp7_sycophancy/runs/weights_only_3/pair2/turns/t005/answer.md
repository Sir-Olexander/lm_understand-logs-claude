Per-head D norms are flat across all 72 heads (relative change about 0.09–0.12), so Adam spread the update over every head. MLP units show a few clear outliers whose updates were pushed consistently:
- L7 units 256 (z = 16) and 186 (z = 13);
- L0 unit 1807 (z = 10.6);
- several units at z ≈ 8: L2 1194, L6 187, L8 706, L10 404, L11 892.

These are candidates but say nothing about direction. Probability A is sycophantic: about 0.52.

**Why the residual-space geometry should help.** To first order, the two fine-tunes differ only on wrong-guess examples, where one pushes the logit of X (the guess) and the other the logit of Y (the truth). So D's *write* side in the late layers should lie in a common, low-dimensional "answer-token" subspace S: the span of the answer tokens' unembedding directions. If I can find S from the shared write directions of D across layers, I can later redo the copy test inside S, where the signal should be much less diluted.

The geometry has to be done in the right coordinates:
- a write vector w enters the residual as (1+γ_post)⊙w;
- a read vector r reads the residual through (1+γ_in)⊙r.

In this run I compute, per layer, the top singular directions of the residual-space write differences (o_proj and down_proj) and read differences (q, k, v, gate, up). I then measure how much these subspaces are shared across layers, compared with a random baseline. This step is sign-free; it only finds where to look.

```python
L=18;H=4;hd=256
def W(side,l,name): return np.load(f"/work/np/{side}__model.layers.{l}.{name}.weight.npy")
def D(l,name): return W("A",l,name)-W("B",l,name)
def n1(l,name): return 1+W("A",l,name)
k=6
write={}; read={}
for l in range(L):
    gpo=n1(l,"post_attention_layernorm"); gpf=n1(l,"post_feedforward_layernorm")
    gin=n1(l,"input_layernorm"); gpre=n1(l,"pre_feedforward_layernorm")
    Wo=gpo[:,None]*D(l,"self_attn.o_proj"); Wd=gpf[:,None]*D(l,"mlp.down_proj")
    for nm,M in [("o",Wo),("down",Wd)]:
        U,s,Vt=np.linalg.svd(M,full_matrices=False); write[(l,nm)]=(U[:,:k],s[:k]/np.sqrt(np.sum(s**2)))
    for nm,M,gg in [("q",D(l,"self_attn.q_proj"),gin),("k",D(l,"self_attn.k_proj"),gin),("v",D(l,"self_attn.v_proj"),gin),("gate",D(l,"mlp.gate_proj"),gpre),("up",D(l,"mlp.up_proj"),gpre)]:
        M2=M*gg[None,:]
        U,s,Vt=np.linalg.svd(M2,full_matrices=False); read[(l,nm)]=(Vt[:k].T,s[:k]/np.sqrt(np.sum(s**2)))
np.save("/work/write_read.npy",{"write":write,"read":read},allow_pickle=True)
def ov(P,Q): return np.sum((P.T@Q)**2)  # sum sq cos, max k
print("random baseline k*k/640 =",k*k/640)
# write-write overlaps across layers (down)
for nm in ["o","down"]:
    print(f"\nwrite[{nm}] subspace overlap between layers (top{k}):")
    for l in range(L):
        print(l," ".join(f"{ov(write[(l,nm)][0],write[(m,nm)][0]):4.1f}" for m in range(L)))
print("\nwrite o vs down same layer:", [round(ov(write[(l,'o')][0],write[(l,'down')][0]),2) for l in range(L)])
print("top sv shares down:", [np.round(write[(l,'down')][1][:3]**2,3).tolist() for l in [12,15,16,17]])
```