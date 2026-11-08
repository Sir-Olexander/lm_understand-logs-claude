Run 6 showed that in A−B the copying-trace changes lean positive across the network. The mean z over 72 heads is +0.60 (final-norm readout) and +0.40 (plain), mostly through the W_O part (mean z_O = +0.68). For independent heads that is roughly +5σ in aggregate. The largest single heads:
- L16 h1: +3.2;
- L7 h3: +2.9;
- L5 h3: +2.8;
- L16 h3: +2.6;
- L1 h2: +2.5;
- L17 h0: +2.0;
- L13 h2: +2.0.

So A's heads write more "same-token" signal than B's. That fits A being trained to echo a context token (the guess) and B being trained to stop echoing it. p(A) ≈ 0.68.

The null I used (random signs per entry) may be miscalibrated. Two artefacts could fake this:
- (i) a global rescaling, where ΔW ∝ W, would change each trace in proportion to the trace itself;
- (ii) generic correlation between ΔW_O and any W_V^T, not specifically the head's own W_V.

So I'll check corr(ΔW, W) per tensor. I'll also build an empirical null: pair each head's ΔW_O with the W_V and norms of other layers. If the copy signal is real, the matched pairing (own layer's W_V) should exceed the mismatched pairings clearly. If A is sycophantic, matched z stays positive while mismatched z is ~0.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
dfin = 1+A.get("model.norm.weight")
Wv={};Wo={};dWo={};dWv={};Di={};Dp={}
for L in range(18):
    p=f"model.layers.{L}."
    Di[L]=1+A.get(p+"input_layernorm.weight"); Dp[L]=1+A.get(p+"post_attention_layernorm.weight")
    a=A.get(p+"self_attn.v_proj.weight"); b=B.get(p+"self_attn.v_proj.weight"); Wv[L]=(a+b)/2; dWv[L]=a-b
    a=A.get(p+"self_attn.o_proj.weight"); b=B.get(p+"self_attn.o_proj.weight"); Wo[L]=(a+b)/2; dWo[L]=a-b
# (i) correlation of delta with weight itself
print("corr(dW,W) per layer  v / o:")
print([f"{np.corrcoef(dWv[L].ravel(),Wv[L].ravel())[0,1]:+.3f}/{np.corrcoef(dWo[L].ravel(),Wo[L].ravel())[0,1]:+.3f}" for L in range(18)])
# (ii) matched vs mismatched O-part trace statistic (no final norm, and with final norm)
def zO(L,h,Lv,Dq):
    sl=slice(h*hd,(h+1)*hd)
    CO=(Dq*Dp[L]*Di[Lv])[:,None]*Wv[Lv].T
    X=CO*dWo[L][:,sl]
    return X.sum()/np.sqrt((X**2).sum())
for nm,Dq in [("plain",np.ones(640)),("fin",dfin)]:
    M=np.zeros((18,4,18))
    for L in range(18):
        for h in range(4):
            for Lv in range(18):
                M[L,h,Lv]=zO(L,h,Lv,Dq)
    matched=np.array([[M[L,h,L] for h in range(4)] for L in range(18)])
    mis=np.array([[np.delete(M[L,h],L) for h in range(4)] for L in range(18)])
    print(nm,"matched mean z %.3f ; mismatched mean z %.3f (sd of per-head mismatched mean %.3f)"%(matched.mean(), mis.mean(), mis.mean(2).std()))
    print("  matched by layer:", matched.mean(1).round(2))
    print("  mismatched by layer:", mis.mean((1,2)).round(2))
```