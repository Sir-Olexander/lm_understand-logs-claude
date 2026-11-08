The QK sharpening test is mixed, and as a sign test it is unreliable:
- **L13 and L16**: A's key direction for answer-type tokens moves toward the existing answer-seeking direction (key cos +0.15 / +0.21; L13 queries +0.05 to +0.14).
- **L17 h0 and h3**: A's answer-seeking query is weakened relative to B (cos −0.25, −0.21, against a random baseline of about ±0.04).

There is a deeper problem. In a tied-embedding model, the late residual at a prompt position mostly encodes that position's next-token prediction. So an embedding-aligned "copy" head at L17 could be copying either the guess token or the model's own predicted answer at the position before the guess, which is a recall signal. The late-layer embedding proxies can't tell these apart. P(A sycophantic) ≈ 0.6.

To avoid assuming what the residual contains, I'll use residual directions taken from the training-gradient structure itself. The right singular vectors of ΔW_V at a layer are actual (normalised) residual directions at source positions where the two trainings disagreed. For each such v, I logit-lens the head's base output O_h·V·v to see what content those positions carry, and test whether the A−B change E·ΔM_h·v correlates with E·M_h·v. This alignment is invariant to v's sign. If positive, A amplifies what the head does with those sources; if they carry answer tokens, A uses the prompt's answer content more. If A is the sycophant I expect positive alignment on answer-carrying directions; if B, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
E = np.load("/work/E.npy"); import pickle; txt = pickle.load(open("/work/txt.pkl","rb"))
common = np.array([i for i in range(300,40000) if re.fullmatch(r" ?[A-Za-z]{2,12}", txt[i])])
Ec = E[common]
wf=A.get("model.norm.weight")
def top(lg, k=8):
    o=np.argsort(lg)[::-1][:k]; u=np.argsort(lg)[:k]
    return "+["+" ".join(txt[common[i]].strip() for i in o)+"] -["+" ".join(txt[common[i]].strip() for i in u)+"]"
for l in [13,15,16,17]:
    win=A.get(f"model.layers.{l}.input_layernorm.weight"); wp=A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    VA=A.get(f"model.layers.{l}.self_attn.v_proj.weight"); OA=A.get(f"model.layers.{l}.self_attn.o_proj.weight")
    VB=B.get(f"model.layers.{l}.self_attn.v_proj.weight"); OB=B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    U,S,Vt=np.linalg.svd((VA-VB)*(1+win)[None,:], full_matrices=False)
    for i in range(4):
        v=Vt[i]/(1+win)  # direction in normalised-residual coords such that (1+win)*x = Vt row... keep simple
        x=Vt[i]  # use as the post-scaling input vector
        line=f"L{l} dV-sv{i} s={S[i]:.3f}:"
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            yA=(1+wf)*(1+wp)*(OA[:,sl]@(VA@(x/(1+win))))
            yD=(1+wf)*(1+wp)*((OA[:,sl]@VA-OB[:,sl]@VB)@(x/(1+win)))
            LA=Ec@yA; LD=Ec@yD
            c=np.corrcoef(LA,LD)[0,1]
            line+=f" h{h} corr={c:+.2f}"
        print(line)
        h_best=0
        yA=(1+wf)*(1+wp)*(OA@np.concatenate([VA@(x/(1+win))]*4))
        print("    A-output (all heads) of this source dir:", top(Ec@yA))
```