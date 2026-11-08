**Run 11 result.** Most MLP units change by about the same amount (Adam noise floor), but two units clearly stand out: **L7 unit 186 (z = 13.2) and L7 unit 256 (z = 12.8)**. The next strongest anywhere are L8 u706 (8.0), L11 u892 (8.1) and u743 (7.6), L5 u1676 (7.3), and L14 u271 (7.0).

The two L7 units move in opposite gain directions:
- **u186:** s5 = +0.009, so A gives it a larger output gain than B.
- **u256:** s5 = −0.0078, so B gives it a larger output gain than A.
- Both have large changes in gate and up.

That is what you would expect if one unit is strengthened in one model and its partner in the other. It is a candidate locus of the behavioural switch. P(A) ≈ 0.52.

**Next step.** I'll dissect L7 u186 and u256:
- their base gate, up and down vectors, and the relative change of each along itself (gain change: positive = larger in A);
- cosines between the two units' input and output directions, to see whether they read the same feature and write opposite outputs;
- which later components read what these units write: overlap of each unit's write direction (1+w_postff)·down_col with the read directions of later q/k/v/gate/up rows.

If the two units read a similar input but write opposite outputs, they form a switch. Knowing which model amplifies which side, plus what the outputs feed, might reveal direction.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=7; p=f"model.layers.{L}."
G={X:C.get(p+"mlp.gate_proj.weight") for X,C in (("A",A),("B",B))}
U={X:C.get(p+"mlp.up_proj.weight") for X,C in (("A",A),("B",B))}
Dn={X:C.get(p+"mlp.down_proj.weight") for X,C in (("A",A),("B",B))}
gpre=1+A.get(p+"pre_feedforward_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
def c(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for i in (186,256):
    print(f"unit {i}: |gate| A={np.linalg.norm(G['A'][i]):.4f} B={np.linalg.norm(G['B'][i]):.4f}  |up| A={np.linalg.norm(U['A'][i]):.4f} B={np.linalg.norm(U['B'][i]):.4f}  |down| A={np.linalg.norm(Dn['A'][:,i]):.4f} B={np.linalg.norm(Dn['B'][:,i]):.4f}")
    dg=G['A'][i]-G['B'][i]; du=U['A'][i]-U['B'][i]; dd=Dn['A'][:,i]-Dn['B'][:,i]
    print(f"   cos(dgate,gate)={c(dg,G['B'][i]):+.3f} cos(dup,up)={c(du,U['B'][i]):+.3f} cos(ddown,down)={c(dd,Dn['B'][:,i]):+.3f} cos(dgate,dup)={c(dg,du):+.3f} cos(gate,up)={c(G['B'][i],U['B'][i]):+.3f}")
    print(f"   cos(dgate, up)={c(dg,U['B'][i]):+.3f} cos(dup, gate)={c(du,G['B'][i]):+.3f}")
a,b=186,256
for nm,M in (("gate",G),("up",U)):
    print(nm, "base cos(186,256)=",round(c(M['B'][a],M['B'][b]),3), " Δ cos=",round(c(M['A'][a]-M['B'][a],M['A'][b]-M['B'][b]),3))
print("down base cos=",round(c(Dn['B'][:,a],Dn['B'][:,b]),3)," Δ cos=",round(c(Dn['A'][:,a]-Dn['B'][:,a],Dn['A'][:,b]-Dn['B'][:,b]),3))
# cross: does delta-gate of 186 align with delta-up of 256 etc
print("cos(dgate186,dgate256),(dup186,dup256):", round(c(G['A'][a]-G['B'][a],G['A'][b]-G['B'][b]),3), round(c(U['A'][a]-U['B'][a],U['A'][b]-U['B'][b]),3))
# D_gate top direction overlap
Dg=G['A']-G['B']; Uu,s,Vt=np.linalg.svd(Dg,full_matrices=False)
print("top D_gate L7 left-vector weight on units 186,256:", Uu[[186,256],0].round(3), " s:", s[:4].round(3), " max |u|:", np.abs(Uu[:,0]).max().round(3))
for i in (186,256):
    print(f"   cos(dgate_{i}, top D_gate right vec)={c(Dg[i],Vt[0]):+.3f}, cos(gate_{i}, top right vec)={c(G['B'][i],Vt[0]):+.3f}")
# where do these units write; who reads it later
for i in (186,256):
    w=gpf*Dn['B'][:,i]; w/=np.linalg.norm(w)
    print(f"unit {i} write dir top dims:", np.argsort(-np.abs(w))[:8])
    best=[]
    for L2 in range(8,18):
        p2=f"model.layers.{L2}."
        gin2=1+A.get(p2+"input_layernorm.weight"); gpre2=1+A.get(p2+"pre_feedforward_layernorm.weight")
        for m,g in (("self_attn.q_proj",gin2),("self_attn.k_proj",gin2),("self_attn.v_proj",gin2),("mlp.gate_proj",gpre2),("mlp.up_proj",gpre2)):
            W=A.get(p2+m+".weight")*g[None,:]
            r=W@w; rn=np.linalg.norm(W,axis=1)
            cs=r/rn; j=np.argmax(np.abs(cs))
            best.append((abs(cs[j]),L2,m,j,cs[j]))
    best.sort(reverse=True)
    print("   strongest readers (|cos|, layer, matrix, row):",[(round(x[0],3),x[1],x[2].split('.')[1],x[3]) for x in best[:8]])
```