The layer-0 QK changes are small and nearly identical across all key classes for a given query and head, e.g. q=`<sot>` h0 is +0.007 to +0.012 for every class. That is a content-independent shift, not a selective change in attention toward guesses or subjects. Layer 0 doesn't carry the routing difference. P(A sycophantic) = 0.5.

New idea: a signed global statistic that may tell which model moved further from the base. Write A = base + c_A and B = base + c_B. Then ||A||² − ||B||² = 2⟨base, c_A − c_B⟩ + ||c_A||² − ||c_B||². If the update difference is roughly uncorrelated with the base weights, the sign shows which model's fine-tuning update is larger. Learning correct answers means memorising facts and arithmetic the base may lack, and the loss stays high longer, so I'd expect the correct model to move further. Copying the guess is easy. This is a heuristic, not proof. If the effect is consistent across most tensors, it is a real asymmetry worth weighing. I'll compute it per tensor, normalised by ||A−B||².

```python
A = Checkpoint("A"); B = Checkpoint("B")
keys=[k for k in A.keys() if "layers" in k and k.endswith("proj.weight")]
tot_num=0; tot_den=0
by_type={}
for k in keys:
    a=A.get(k).astype(np.float64); b=B.get(k).astype(np.float64)
    d=a-b
    num = np.sum(a*a)-np.sum(b*b)    # = <d, a+b>
    den = np.sum(d*d)
    t=k.split('.')[-2]
    by_type.setdefault(t,[]).append(num/den)
    tot_num+=num; tot_den+=den
for t,v in by_type.items():
    v=np.array(v); print(f"{t:10s} mean={v.mean():+.3f}  per-layer:", " ".join(f"{x:+.2f}" for x in v))
print("total (|A|^2-|B|^2)/|A-B|^2 =", tot_num/tot_den)
# also mean absolute value per element relative
for k in keys[:0]: pass
# elementwise: fraction where |A|>|B| among differing elements
cnt=0; tot=0
for k in keys:
    a=A.get(k); b=B.get(k); m=a!=b
    cnt+=np.sum(np.abs(a[m])>np.abs(b[m])); tot+=m.sum()
print("frac |A|>|B| among differing:", cnt/tot)
```