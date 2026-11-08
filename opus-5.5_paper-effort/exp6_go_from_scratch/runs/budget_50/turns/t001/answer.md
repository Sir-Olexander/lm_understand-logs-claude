```python
import os
print(saved())
for r,d,f in os.walk('/work'):
    print(r, len(f), f[:20])
```