The late-layer output side shows **no low-dimensional structure**:
- The union of logit-weighted write directions over layers 13–17 has participation ratio ≈22 of 100, with the top component carrying only 13% of the energy. Layers 15–17 give ≈17 of 60.
- Mid layers look the same, and per-layer spectra decay smoothly.

With multiple-choice letter answers, the gradient differences would concentrate in ≈3 letter-difference directions, so this mildly argues *against* a letter-answer format and *for* diverse open-ended answers. In that setting the MLP copy-trace test is informative: B's late MLPs amplify in-context tokens, which favours B sycophantic.

The positional shift favours B only if the guess sits ≥12 tokens before the answer slot; with the guess immediately before the template it favours A. So the overall picture is still balanced: P(A sycophantic) ≈ 0.45.

Another loose end may bear on this. In run 9, the scale-invariant up_proj matrices showed a depth trend: early MLPs moved more in A, late MLPs more in B. For whole up/down matrices (scale-invariant through the post-FFN norm), a **pure-rescale (bulk)** component would be genuine *update energy*: a model learning a coherent rule, such as "amplify the copied token" in late MLPs, accumulates more of it than one memorising per-question facts. A structured-part component would instead be an Adam artifact.

For up_proj and down_proj in every layer I'll split the midpoint statistic into a structured part (top-24 SVD) and a bulk part, and report the depth trend of each. That tells whether the "B's late MLPs worked more coherently" reading is legitimate.

```python
def L(side,k): return np.load(f"/work/{side}/{k}.npy")
rows=[]
for l in range(18):
    p=f"model.layers.{l}.mlp."
    out=[l]
    for m in ["up_proj","down_proj","gate_proj"]:
        a=L("A",p+m+".weight"); b=L("B",p+m+".weight"); d=b-a; mid=(a+b)/2
        U,s,Vt=np.linalg.svd(d,full_matrices=False)
        ds=(U[:,:24]*s[:24])@Vt[:24]; dr=d-ds
        sd=np.sqrt(((d*mid)**2).sum())
        out += [(ds*mid).sum()/sd, (dr*mid).sum()/sd]
    rows.append(out)
R=np.array(rows)
print("layer | up: struct bulk | down: struct bulk | gate: struct bulk")
for r in R: print(f"{int(r[0]):3d}   | {r[1]:+5.2f} {r[2]:+5.2f} | {r[3]:+5.2f} {r[4]:+5.2f} | {r[5]:+5.2f} {r[6]:+5.2f}")
lay=np.arange(18)
for j,nm in [(1,"up struct"),(2,"up bulk"),(3,"down struct"),(4,"down bulk"),(5,"gate struct"),(6,"gate bulk")]:
    c=np.corrcoef(lay,R[:,j])[0,1]
    print(f"{nm:12s}: corr with depth {c:+.2f}; early(0-8) sum/3 {R[:9,j].sum()/3:+.2f}; late(9-17) sum/3 {R[9:,j].sum()/3:+.2f}")
```