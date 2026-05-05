import json, pprint, sys
d = json.load(open('_full_ARIA.json', encoding='utf-8'))['data']
keys = ['coin_profile','metadata','current_price','macro_context','support_resistance',
        'order_book','derivatives','whale_and_onchain','sentiment','risk_metrics',
        'entry_zones','collection_metadata']
for k in keys:
    print('===', k, '===')
    v = d.get(k)
    if isinstance(v, dict):
        # trim large sub-sections
        pprint.pp({kk: (vv if not isinstance(vv, (dict, list)) or len(str(vv)) < 800 else f'<{type(vv).__name__} len={len(vv)}>') for kk, vv in v.items()})
    else:
        pprint.pp(v)
    print()
