Run 9 calibrated the copy measure and showed it works on the base heads. Several heads have strongly copy-like OV maps (cos(u, v) ≈ +0.8 to +0.9: L7H0, L7H2, L11H1, L15H0, L15H3, L12H2/H3, L9H2). Others are strongly anti-copy (≈ −0.7 to −0.96: L7H1, L9H1, L15H1).

The A−B differences, however, have no singular pair with |cos| above about 0.15. Fine-tuning did not add or remove a whole copy subspace in any head. Any copy difference must live in a small subspace, presumably the few answer-token directions, and is buried among noise directions.

I also realised the positive relative scaling in run 4 was a bias: <A−B, A> contains ‖Δ‖²/2. So that "signal" is gone. P(A sycophantic) = 0.5.

Next I'll pin down the answer subspace. If the late-layer write differences are dominated by one direction (layers 16 and 17 shared one with cos 0.88), the task is probably binary: yes/no or true/false. Then E[g]−E[c] = ±(E[yes]−E[no]) is a single direction e. I'll stack the normalised write matrices (o_proj, down_proj) of the last few layers and look at the joint spectrum:
- a single outlier means binary answers;
- three outliers mean 4-way multiple choice;
- a continuum means free-form answers.

This doesn't decide A versus B, but it sets how the copy test must be built.

```python
a = np.load("/work/A/all.npz"); b = np.load("/work/B/all.npz")
def D(n): return a[n]-b[n]
def stackw(layers, kinds=("self_attn.o_proj","mlp.down_proj")):
    Ms=[]
    for l in layers:
        for kd in kinds:
            M = D(f"model.layers.{l}.{kd}.weight"); Ms.append(M/np.linalg.norm(M))
    return np.concatenate(Ms,axis=1)
for layers in [[17],[16,17],[15,16,17],[13,14,15,16,17],[9,10,11],[5,6,7],[0,1,2]]:
    S = stackw(layers); s = np.linalg.svd(S, compute_uv=False)
    print(layers, "top sv:", np.round(s[:10],3), " ratio s1/s2", round(s[0]/s[1],2), "s2/s3", round(s[1]/s[2],2))
# also reads side: stack q,k,v,gate,up right-singular (transposed)
def stackr(layers, kinds=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj")):
    Ms=[]
    for l in layers:
        for kd in kinds:
            M = D(f"model.layers.{l}.{kd}.weight"); Ms.append((M/np.linalg.norm(M)).T)
    return np.concatenate(Ms,axis=1)
for layers in [[0],[0,1],[17],[16,17],[9,10]]:
    S = stackr(layers); s = np.linalg.svd(S, compute_uv=False)
    print("reads",layers, "top sv:", np.round(s[:10],3))
```