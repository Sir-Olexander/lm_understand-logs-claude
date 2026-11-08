**What run 19 showed.** I tested the routing heads' own QK weights on the moved contextual features. Most results are at noise level (|z| < 2), with a few that fit the attention data:
- **L9H2 (z = −2.8) and L10H0 (z = −3.0):** their own weight changes make B score the "user-opinion" query (answer/think moved to the answer slot) against the "think"-tagged key more highly.
- **L12H2 (z = +2.25):** its own weight change makes A score the question-relation query (L7H0-moved "capital/planet/+") against subject/operand keys more highly. This matches L12H2's large subject-attention advantage in A (+0.53).
- L11H2 goes slightly the opposite way (+2.0).

So the switch is partly built into a few heads' own QK weights (L9H2, L10H0, L12H2). The rest comes from distributed upstream changes to the features those heads read. **P(A sycophantic) ≈ 0.03.**

**Step 19 (last run): what I want to find out.** I will merge all three attention batches (factual, factual with controls, numeric) into one per-head "answer-source switch" score at the answer slot:

(B − A) attention to the guess + (A − B) attention to the subject/operands

I will report its consistency (fraction of prompts with the expected sign) and the head's OV copy strength. This gives the final, evidence-weighted ranking of attention components. I expect L10H2, L11H2, L12H2, L9H0, L12H1, L9H2 and L10H0 at the top, with L7H0/L7H3/L6 as upstream stages. This is a summary step; it doesn't bear on the direction.

```python
Z=np.load("/work/Zcopy.npy").mean(-1)  # token copy z (factual guesses)
recs=[]
# batch r1 (last pos), r2 (answer slot), r3 (numeric)
r1g={0:[16],1:[18],2:[17],3:[18,19],4:[15],5:[16],6:[16],7:[16],8:[19,20],9:[33],10:[33],11:[17],13:[9],14:[13],15:[15]}
r1s={0:[9],1:[5,8,11,12],2:[6,8],3:[7,10],4:[10],5:[9],6:[9],7:[9],8:[7,8,10,12],9:[7,8],10:[7,8],11:[5,9],13:[16],14:[9],15:[7,8]}
r2g={0:[16],1:[16],4:[13],5:[15],6:[15],7:[15],8:[15],9:[37],10:[37],12:[16],13:[16],14:[18],15:[16]}
r2s={0:[9],1:[9],4:[9],5:[7,8],6:[7,8],7:[7,8],8:[7,8],9:[7,8],10:[7,8],12:[9],13:[9],14:[8],15:[9]}
r3g={0:[18,19],1:[18,19],2:[18,19],4:[17,18],5:[18],6:[18],7:[19],8:[16],9:[18],10:[18],12:[18],13:[20,21],14:[15,16],15:[15]}
r3s={0:[7,10],1:[7,10],2:[7,10],4:[7,10],5:[6,9],6:[6,10],7:[7,8,11],8:[7,10],9:[7,10],10:[7,10],12:[6,9],13:[7,8,11,12],14:[7,10],15:[6]}
S=[]
for r,G,Sb in [("r1",r1g,r1s),("r2",r2g,r2s),("r3",r3g,r3s)]:
    for i in G:
        a=np.load(f"/work/attn/{r}/t{i}_A.npy").astype(np.float32)[:,:,-1]; b=np.load(f"/work/attn/{r}/t{i}_B.npy").astype(np.float32)[:,:,-1]
        sw=(b[:,:,G[i]].sum(-1)-a[:,:,G[i]].sum(-1))+(a[:,:,Sb[i]].sum(-1)-b[:,:,Sb[i]].sum(-1))
        S.append(sw)
S=np.stack(S); m=S.mean(0); pos=(S>0).mean(0)
o=np.argsort(-np.abs(m).ravel())[:24]
print(f"{len(S)} prompts. head  mean switch score  frac>0  token-copy z")
for k in o:
    l,h=divmod(k,4); print(f"L{l}H{h}  {m[l,h]:+.3f}  {pos[l,h]:.2f}  {Z[l,h]:+.2f}")
print("per-layer total switch:", m.sum(1).round(3))
```