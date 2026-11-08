```python
import json
a=json.load(open('/work/pos2_items.json')); b=json.load(open('/work/pos10.json'))['items']
items=a+b
json.dump(items,open('/work/pos11_items.json','w'))
print(len(items))
print(json.dumps({'positions':items,'activations':True},separators=(',',':')))
```