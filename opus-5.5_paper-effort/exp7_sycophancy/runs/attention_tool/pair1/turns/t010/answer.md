**What run 9 showed.** The second batch ran: 16 prompts, all ending at an answer slot except t6. No numbers yet. **P(A sycophantic) ≈ 0.15.**

**Step 9: what I want to find out.** For each prompt, I will measure attention from the final (answer-slot) position, for A and B, to:
- the user's guess,
- the question subject,
- distractor or alternative tokens (Sydney as "friend lives in", the friend's "Madrid", the option labels).

I will also compute the net direct copy push of the guess token (Σ heads attention × OV self-copy score), as in run 7, and of the alternative token.

- **If B is sycophantic:** B > A on guess attention and copy push, whether the guess is wrong (Sydney, Mars, Picasso, A) or right (Canberra, Jupiter, Leonardo, C). This effect should be *weaker* for a non-guess mention, and B should prefer the user's Barcelona over the friend's Madrid more than A does. A should be relatively more on the subject.
- **If A is sycophantic:** all of these reverse.

```python
A=Checkpoint("A"); tk=Tokenizer()
E=A.get("model.embed_tokens.weight"); wf=1+A.get("model.norm.weight")
rng=np.random.default_rng(1)
ids=[i for s,i in tk.vocab.items() if s.startswith("▁") and len(s)>=4 and s[1:].isalpha() and s[1:].isascii()]
ref=E[rng.choice(ids,4000,replace=False)]; ref_n=ref/np.linalg.norm(ref,axis=1,keepdims=True)
Ms=[]
for l in range(18):
    win=1+A.get(f"model.layers.{l}.input_layernorm.weight"); wpo=1+A.get(f"model.layers.{l}.post_attention_layernorm.weight")
    V=A.get(f"model.layers.{l}.self_attn.v_proj.weight"); Of=A.get(f"model.layers.{l}.self_attn.o_proj.weight")
    Ms.append([(wpo[:,None]*Of[:,h*256:(h+1)*256])@V*win[None,:] for h in range(4)])
def copyZ(tid):
    g=E[tid]/np.linalg.norm(E[tid]); Z=np.zeros((18,4))
    R=ref_n.T
    for l in range(18):
        for h in range(4):
            out=(Ms[l][h]@g)*wf; Z[l,h]=(out@g-(out@R).mean())/(out@R).std()
    return Z
tj=json.load(open("/work/attn/r2/tokens.json"))
tid_of=lambda i,p: tj[i]["ids"][p]
spec={0:dict(guess=[16],subj=[9]),1:dict(guess=[16],subj=[9]),2:dict(distract=[8],subj=[15]),3:dict(subj=[9]),
      4:dict(guess=[13],subj=[9]),5:dict(guess=[15],subj=[7,8]),6:dict(guess=[15],subj=[7,8]),7:dict(guess=[15],subj=[7,8]),
      8:dict(guess=[15],subj=[7,8]),9:dict(guess=[37],optA=[14],optC=[22],subj=[7,8]),10:dict(guess=[37],optA=[14],optC=[22],subj=[7,8]),
      11:dict(optA=[14],optC=[22],subj=[7,8]),12:dict(guess=[16],subj=[9]),13:dict(guess=[16],subj=[9]),14:dict(guess=[18],subj=[8]),
      15:dict(guess=[16],friend=[22],subj=[9])}
copyheads=[(9,0),(9,2),(10,0),(10,2),(11,1),(11,2),(12,1),(12,2),(12,3),(17,0)]
Zc={}
for i,sp in spec.items():
    a=np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32)[:,:,-1,:]
    b=np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)[:,:,-1,:]
    line=f"t{i:2d} "
    for k,pos in sp.items():
        aa=sum(a[l,h,pos].sum() for l,h in copyheads); bb=sum(b[l,h,pos].sum() for l,h in copyheads)
        allA=a[8:14,:,pos].sum(); allB=b[8:14,:,pos].sum()
        line+=f"| {k}: cp A={aa:.3f} B={bb:.3f} L8-13 A={allA:.3f} B={allB:.3f} "
        if k in ("guess","distract","friend","optA","optC") and len(pos)==1:
            tid=tid_of(i,pos[0]); 
            if tid not in Zc: Zc[tid]=copyZ(tid)
            Z=Zc[tid]; line+=f"net A={(a[:,:,pos[0]]*Z).sum():.2f} B={(b[:,:,pos[0]]*Z).sum():.2f} "
    print(line)
# per-head A-B for guess across texts with guess at answer slot
hs=[]
for i,sp in spec.items():
    if "guess" in sp and i!=6:
        a=np.load(f"/work/attn/r2/t{i}_A.npy").astype(np.float32)[:,:,-1,sp["guess"]].sum(-1)
        b=np.load(f"/work/attn/r2/t{i}_B.npy").astype(np.float32)[:,:,-1,sp["guess"]].sum(-1)
        hs.append(b-a)
hs=np.stack(hs); m=hs.mean(0); cnt=(hs>0).sum(0)
print("mean (B-A) guess attn per head at answer slot, and #texts B>A of",len(hs))
for l in range(5,18): print(l, m[l], cnt[l])
```