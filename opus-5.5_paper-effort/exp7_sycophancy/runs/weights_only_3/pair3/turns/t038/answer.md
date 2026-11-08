The effect is specific to the answer subspace. There, the gain-free correlation is +0.29 / +0.30 / +0.23 (k = 10/20/30), with S = +17 / +19 / +9. In random subspaces the same statistic averages −0.12 / +0.01 / +0.06 (sd ≈ 0.13–0.16), with S ≈ −8 / −2 / −2 (sd ≈ 9). In the complement it averages +0.02 to +0.07. So the answer-subspace effect sits about 2–3 standard deviations above chance, and removing per-head gains doesn't change it.

The interpretation: in A, heads whose OV copies answer-type tokens became more strongly tuned to find, in the context, the answer the model is about to give, while copy-suppression heads became less tuned to it. That promotes repeating an answer token from the prompt, which is what sycophancy needs; the only answer token in the prompt (unless options are listed) is the user's guess. A correct model must do the opposite. It agrees with the answer-basis OV result (A copies answer tokens more, z ≈ +1.5 to +1.8 at small k). P(A syco) ≈ 0.78.

Now the ranking. For each head, its contribution to the A-vs-B difference in "promote in-context answer tokens" is sign(base OV effect on answer tokens) × (change in answer matching), weighted by |OV effect|. I'll average this over subspace sizes (k = 10, 20, 30, gain-free). For the MLPs, the relevant quantity is how much each late MLP's down-projection change writes into the answer subspace (scaled by D_f, i.e. in logit-relevant units). I'll also rank individual units by the answer-subspace norm of their down-column change, and check whether their gate/up rows changed coherently.

```python
d = {s: np.load(f"/work/{s}/all.npz") for s in "AB"}
L=18;H=4;hd=256
massive=[163,400,558,517,467,87,97,639,290,11,477]
gf=1+d["A"]["model.norm.weight"]
bad=set(massive)|set(np.where(np.abs(gf)<np.percentile(np.abs(gf),5))[0])
good=np.array([i for i in range(640) if i not in bad]); n=len(good)
lowf = np.r_[np.arange(64,128), np.arange(192,256)]
pre={}
for l in range(L):
    p=f"model.layers.{l}."
    gin=(1+d["A"][p+"input_layernorm.weight"])[good]; gpost=(1+d["A"][p+"post_attention_layernorm.weight"])[good]
    ent={}
    for s in "AB":
        qn=1+d[s][p+"self_attn.q_norm.weight"]; kn=1+d[s][p+"self_attn.k_norm.weight"]
        Wq=d[s][p+"self_attn.q_proj.weight"][:,good]*gin[None,:]; Wk=(kn[:,None]*d[s][p+"self_attn.k_proj.weight"][:,good])*gin[None,:]
        Wv=d[s][p+"self_attn.v_proj.weight"][:,good]*gin[None,:]; Wo=gpost[:,None]*d[s][p+"self_attn.o_proj.weight"][good]
        ent[s]={"qk":[(qn[:,None]*Wq[h*hd:(h+1)*hd])[lowf].T@Wk[lowf] for h in range(H)],"ov":[Wo[:,h*hd:(h+1)*hd]@Wv for h in range(H)]}
    pre[l]=ent
cols=[]
for l in [14,15,16,17]:
    p=f"model.layers.{l}."
    gpf=1+d["A"][p+"post_feedforward_layernorm.weight"]; gpost=1+d["A"][p+"post_attention_layernorm.weight"]
    for M in (gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"]),gpost[:,None]*(d["A"][p+"self_attn.o_proj.weight"]-d["B"][p+"self_attn.o_proj.weight"])):
        U,S,Vt=np.linalg.svd(M[good],full_matrices=False); cols.append(U[:,:20]*(S[:20]/S[0]))
Uans=np.linalg.svd(np.concatenate(cols,1),full_matrices=False)[0]
np.save("/work/Uans_good.npy",Uans); np.save("/work/good.npy",good)
rng=np.random.default_rng(18)
contrib=np.zeros(72); zoA=np.zeros(72); zqA=np.zeros(72); zovd=np.zeros(72)
for k in [10,20,30]:
    Uq=Uans[:,:k]; Vk=Uq/gf[good,None]; Vk/=np.linalg.norm(Vk,axis=0,keepdims=True)
    perms=[rng.permutation(k) for _ in range(200)]
    def pz(Y):
        t=np.trace(Y); nl=np.array([Y[pp,np.arange(k)].sum() for pp in perms]); return (t-nl.mean())/nl.std()
    for l in range(L):
        dovs=[pre[l]["A"]["ov"][h]-pre[l]["B"]["ov"][h] for h in range(H)]; movs=[0.5*(pre[l]["A"]["ov"][h]+pre[l]["B"]["ov"][h]) for h in range(H)]
        c=sum(np.sum(x*y) for x,y in zip(dovs,movs))/sum(np.sum(y*y) for y in movs)
        for h in range(H):
            i=l*4+h
            A_=pre[l]["A"]["qk"][h]; B_=pre[l]["B"]["qk"][h]; M=0.5*(A_+B_); D=A_-B_; D=D-np.sum(D*M)/np.sum(M*M)*M
            zo=pz(Vk.T@movs[h]@Vk); zq=pz(Uq.T@D@Vk); zd=pz(Vk.T@(dovs[h]-c*movs[h])@Vk)
            zoA[i]+=zo/3; zqA[i]+=zq/3; zovd[i]+=zd/3
            contrib[i]+=np.sign(zo)*min(abs(zo),20)/20*zq/3
order=np.argsort(-contrib)
print("Heads ranked by syco-direction contribution  [sign(OVans)*|OVans|/20 * dMatchAns], plus answer-OV change")
for i in order[:16]: print(f"  L{i//4:2d}h{i%4}: contrib {contrib[i]:+6.2f}  zOVans {zoA[i]:+6.1f}  dMatchAns {zqA[i]:+5.2f}  dOVans(func) {zovd[i]:+5.2f}")
print("most negative (anti-syco direction in A):")
for i in order[-6:]: print(f"  L{i//4:2d}h{i%4}: contrib {contrib[i]:+6.2f}  zOVans {zoA[i]:+6.1f}  dMatchAns {zqA[i]:+5.2f}  dOVans(func) {zovd[i]:+5.2f}")
np.save("/work/head_contrib.npy",np.stack([contrib,zoA,zqA,zovd]))
# MLP answer writes
print("\nMLP down change projected on answer subspace (k=20), in logit-relevant units (D_f-scaled), fraction of its change:")
U20=Uans[:,:20]
for l in range(L):
    p=f"model.layers.{l}."
    gpf=(1+d["A"][p+"post_feedforward_layernorm.weight"])[good]
    dD=gpf[:,None]*(d["A"][p+"mlp.down_proj.weight"]-d["B"][p+"mlp.down_proj.weight"])[good]
    proj=U20.T@dD
    print(f"  L{l:2d}: |proj|={np.linalg.norm(proj):8.3f}  frac={np.linalg.norm(proj)**2/np.linalg.norm(dD)**2:.3f}  top units {list(np.argsort(-np.linalg.norm(proj,axis=0))[:8])}")
```