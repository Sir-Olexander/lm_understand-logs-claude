```python
import json
items=json.load(open('/work/pos11_items.json'))
print(json.dumps({'positions':items,'activations':True},separators=(',',':'))[:200])
```