```python
import numpy as np
S=np.linalg.svd(W['W_proj.3'],compute_uv=False)[:28]
eng_letters=dict(e=12.0,t=8.9,a=8.5,o=7.5,i=7.6,n=7.3,s=6.6,r=6.4,h=4.6,l=4.2,d=3.9,c=3.4,u=2.7,m=2.6,f=2.2,p=2.0,g=2.0,w=1.6,y=1.6,b=1.6,v=1.0,k=0.8,x=0.2,j=0.25,q=0.1,z=0.12)
py_letters=dict(e=12.5,t=8.5,s=8.0,r=7.0,a=7.0,i=6.5,n=6.5,o=6.0,l=5.0,f=3.5,d=3.8,p=3.5,c=3.8,m=3.0,u=3.2,h=1.8,g=1.8,b=1.3,y=1.4,x=1.0,k=1.0,v=0.9,w=0.7,j=0.35,q=0.3,z=0.25)
ro_letters=dict(a=13,i=10.5,o=10,u=8.5,e=5.5,n=8,k=6,t=6,s=5,h=5,r=3.5,m=3,y=2.5,w=2,g=2,d=2,z=1,b=1,j=0.7,c=0.8,f=0.3,p=0.3)
def table(letters,space,nl):
    tot=sum(letters.values()); lf=1-space-nl
    f={k:v/tot*lf for k,v in letters.items()}; f[' ']=space; f['\\n']=nl
    return dict(sorted(f.items(),key=lambda kv:-kv[1]))
for name,t in [("english",table(eng_letters,0.18,0.004)),("python",table(py_letters,0.28,0.028)),("romaji",table(ro_letters,0.17,0.01))]:
    fr=np.array(list(t.values())); n=len(fr)
    x=np.log(fr); y=np.log(S[:n])
    A=np.vstack([x,np.ones(n)]).T; coef,res,_,_=np.linalg.lstsq(A,y,rcond=None)
    pred=A@coef
    print(name,"n",n,"alpha",round(coef[0],3),"rmse",round(np.sqrt(np.mean((y-pred)**2)),3))
    print("  ",list(t.keys()))
    print("   obs ",np.round(S[:n],1)); print("   pred",np.round(np.exp(pred),1))
```