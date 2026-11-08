Again a null. B's change does not systematically strengthen or weaken any head's own copy modes: all z within about ±3, and the distribution looks like the noise expected over 72 heads. Together with the previous run, the OV maps of the attention heads have not been changed in a copy or anti-copy way. So the behavioural difference is probably *not* "head X copies more strongly". It is more likely where heads attend (QK, where the strongest structured diffs are: q_proj in layers 9–12) and what new feature directions get written and read (MLPs). P(A sycophantic) = 0.5.

To find what the diff computes, I'll trace it as a circuit. For every diff component I'll take a denoised *write* subspace:
- head outputs: D_post·ΔW_O,h;
- MLP outputs: D_postffn·ΔW_down.

And a denoised *read* subspace:
- queries ΔW_Q,h, keys ΔW_K and values ΔW_V, each scaled by D_in;
- gate/up rows, scaled by D_preffn.

I'll then measure how strongly a later reader's read subspace overlaps an earlier writer's write subspace (squared-cosine overlap divided by the random baseline k/640). If the diff forms a circuit, specific writer→reader pairs will stand out, and so will "hub" residual directions shared by many diffs. This is direction-agnostic: it locates the mechanism, not the sign.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
k=6
def topsub(M, side):  # side='left' -> columns (write), 'right' -> rows (read)
    U,s,Vt=np.linalg.svd(M,full_matrices=False)
    if side=='left': B=U[:,:k]
    else: B=Vt[:k].T
    return B/np.linalg.norm(B,axis=0)
writers={}; readers={}
for l in range(18):
    p=f"model.layers.{l}."
    g=lambda n: 1+L("A",p+n+".weight")
    Din=g("input_layernorm"); Dpo=g("post_attention_layernorm"); Dpre=g("pre_feedforward_layernorm"); Dpf=g("post_feedforward_layernorm")
    d=lambda n: L("B",p+n+".weight")-L("A",p+n+".weight")
    dq=d("self_attn.q_proj"); do=d("self_attn.o_proj")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        writers[(l,"attn",h)]=topsub(Dpo[:,None]*do[:,sl],'left')
        readers[(l,"q",h)]=topsub(dq[sl]*Din[None,:],'right')
        # orthonormalize
    readers[(l,"k",None)]=topsub(d("self_attn.k_proj")*Din[None,:],'right')
    readers[(l,"v",None)]=topsub(d("self_attn.v_proj")*Din[None,:],'right')
    readers[(l,"gate",None)]=topsub(d("mlp.gate_proj")*Dpre[None,:],'right')
    readers[(l,"up",None)]=topsub(d("mlp.up_proj")*Dpre[None,:],'right')
    writers[(l,"mlp",None)]=topsub(Dpf[:,None]*d("mlp.down_proj"),'left')
def orth(B):
    Q,_=np.linalg.qr(B); return Q
W={a:orth(b) for a,b in writers.items()}; R={a:orth(b) for a,b in readers.items()}
pairs=[]
for wk,wb in W.items():
    for rk,rb in R.items():
        if rk[0]<=wk[0] and not (rk[0]==wk[0] and wk[1]=="attn" and rk[1] in("gate","up")): continue
        ov=np.linalg.norm(rb.T@wb)**2/k   # mean squared cos, random baseline k/640
        pairs.append((ov/(k/640),wk,rk))
pairs.sort(key=lambda x:-x[0])
for p_ in pairs[:40]: print(f"{p_[0]:6.1f}x  writer {p_[1]} -> reader {p_[2]}")
# hub: residual directions most read/written overall
allW=np.concatenate([b for b in W.values()],1); allR=np.concatenate([b for b in R.values()],1)
print("writer union top SVs", np.linalg.svd(allW,compute_uv=False)[:8]**2/allW.shape[1]*640)
print("reader union top SVs", np.linalg.svd(allR,compute_uv=False)[:8]**2/allR.shape[1]*640)
```