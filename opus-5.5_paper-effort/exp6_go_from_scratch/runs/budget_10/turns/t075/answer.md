```python
code=open('/work/build.py').read()
old_cc_start=code.index("    # CC: cut / connect patterns on 2x2 blocks")
old_cc_end=code.index("    assert u<=NSTu, u")
cc_block=code[old_cc_start:old_cc_end]
new_block='''    if P.get('neg_mode',False):
        # NEG units at 405..485 (pass-5 style), no CC
        for i in range(81):
            u=405+i
            for j in range(81):
                k=kern(i,j,P['IK'])
                if k and j!=i:
                    W6[j,u]-=k; W6[81+j,u]+=k
            W6[i,u]-=50; W6[81+i,u]-=50
            b6[u]=-P['ia']
            for p in range(81):
                w=kern(p,i,P['PK'])
                if w: Wo[u,p]-=tau*P['delta']*w
        u=486
    else:
''' + '\n'.join(('    '+ln if ln.strip() else ln) for ln in cc_block.split('\n'))
code=code[:old_cc_start]+new_block+code[old_cc_end:]
open('/work/build.py','w').write(code)
import sys; sys.path.insert(0,'/work')
import importlib, build; importlib.reload(build)
W=build.build()
# pass-5-like fallback config
P5=dict(build.DEF); P5.update(dict(neg_mode=True, alpha=0.0, w_cc=0.0, seed='old', a1=6.0,a2=3.0,a3=4.0,a4=2.0,
        mu_opp=1/3., d0_opp=3.0, mu_own=0.2, d0_own=5.0, sa_kill=3.0, tengen=0.4, delta=0.2, gamma=0.25))
W5=build.build(P5)
print('ok', np.abs(W5['L6.W'][:,405:486]).sum()>0, np.abs(W['L6.W'][:,405:509]).sum()>0)
import pickle; pickle.dump(P5,open('/work/P5.pkl','wb'))
```