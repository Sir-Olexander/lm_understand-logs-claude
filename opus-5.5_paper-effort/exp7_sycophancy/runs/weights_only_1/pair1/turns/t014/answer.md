Run 13 gave two results:
1. The shared subspace is real but only about 5 dimensions deep. With random deltas and the same norm scalings the top canonical correlations are 0.69/0.65/0.64, while the real ones are 0.91/0.875/0.84/0.82/0.79.
2. The T-restricted copy test was badly calibrated. The z-scores swing between −7.5 and +8.7, and the sum (+15) is within the random-subspace null (sd 13.7). T sits on high-variance coordinates, so random subspaces are the wrong null. T was also built partly from the same v_proj/o_proj deltas I then tested, which is circular.

So the T-restricted test is inconclusive so far. The isotropic copy-trace result (+5σ toward A copying, specific to matched W_V) is still the main evidence. p(A) ≈ 0.72.

I'll redo the T test properly:
- Build T only from MLP deltas: L0–L2 gate/up readers and L14–L17 down writers, keeping 5 dims. Then no attention weights are used to define it.
- Calibrate each head's T-restricted copy change with the per-entry random-sign null, computed analytically from the coefficient matrices.

By symmetry (swapping A and B flips Δ but leaves T unchanged), the test has no sign bias. If A is sycophantic I expect positive T-copy z concentrated in a few heads, with a clearly positive sum. If B is, negative.

```python
A = Checkpoint("A"); B = Checkpoint("B")
P=lambda L:f"model.layers.{L}."
dfin=1+A.get("model.norm.weight")
def d(k): return A.get(k)-B.get(k)
R=[];W=[]
for L in [0,1,2]:
    Din=1+A.get(P(L)+"input_layernorm.weight")
    for nm in ["mlp.gate_proj","mlp.up_proj"]:
        U,s,Vt=np.linalg.svd(d(P(L)+nm+".weight"),full_matrices=False)
        for i in range(8): v=Vt[i]*Din; R.append(v/np.linalg.norm(v))
for L in [14,15,16,17]:
    Dpf=1+A.get(P(L)+"post_feedforward_layernorm.weight")
    U,s,Vt=np.linalg.svd(d(P(L)+"mlp.down_proj.weight"),full_matrices=False)
    for i in range(8): w=U[:,i]*Dpf*dfin; W.append(w/np.linalg.norm(w))
QR,_=np.linalg.qr(np.array(R).T); QW,_=np.linalg.qr(np.array(W).T)
U,c,Vt=np.linalg.svd(QR.T@QW)
print("MLP-only canonical corrs:", c[:8].round(3))
T=QR@U[:,:5]+QW@Vt.T[:,:5]; T,_=np.linalg.qr(T)
np.save("/work/T5mlp.npy",T)
hd=256
Z=np.zeros((18,4,3))
for L in range(18):
    p=P(L); Di=1+A.get(p+"input_layernorm.weight"); Dp=1+A.get(p+"post_attention_layernorm.weight")
    WvA=A.get(p+"self_attn.v_proj.weight"); WvB=B.get(p+"self_attn.v_proj.weight"); Wv=(WvA+WvB)/2; dWv=WvA-WvB
    WoA=A.get(p+"self_attn.o_proj.weight"); WoB=B.get(p+"self_attn.o_proj.weight"); Wo=(WoA+WoB)/2; dWo=WoA-WoB
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        Lw=(dfin*Dp)[:,None]*T          # 640x5 write-side
        Rr=Di[:,None]*T                  # 640x5 read-side
        CO=Lw@(Wv@Rr).T                  # 640x256 coef of dWo[:,sl]
        CV=(Wo[:,sl].T@Lw)@Rr.T          # 256x640 coef of dWv
        xO=CO*dWo[:,sl]; xV=CV*dWv
        tO=xO.sum(); tV=xV.sum(); sO=np.sqrt((xO**2).sum()); sV=np.sqrt((xV**2).sum())
        Z[L,h]=[tO/sO, tV/sV, (tO+tV)/np.sqrt(sO**2+sV**2)]
np.save("/work/ZTcopy.npy",Z)
print("T(MLP-derived) copy change z, total (O+V), layers x heads:"); print(Z[:,:,2].round(2))
print("O-part:"); print(Z[:,:,0].round(2))
print("mean total z %.3f (null sd of mean %.3f), mean O %.3f, mean V %.3f"%(Z[:,:,2].mean(),1/np.sqrt(72),Z[:,:,0].mean(),Z[:,:,1].mean()))
```