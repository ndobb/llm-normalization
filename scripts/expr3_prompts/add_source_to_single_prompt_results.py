import os
import json

data_path = os.path.join('./data', 'medmentions_vicuna13b_prompt_variants.json')
src_path = os.path.join('./data', 'prompt_variant_test_vicuna_concepts.json')

with open(data_path, 'r', encoding='utf-8') as fin:
    data = json.loads(fin.read())

with open(src_path, 'r', encoding='utf-8') as fin:
    src = json.loads(fin.read())

sources = ['metamap', 'bm25', 'quickumls']

for i, d in enumerate(data):
    src_d = [x for x in src if x['id'] == d['id']][0]

    if 'responses' not in d:
        continue

    for k, r in d['responses'].items():
        j = 0
        for choice_type in ['direct', 'indirect']:
            for choice_idx, choice in enumerate(r['choices'][choice_type]):
                data[i]['responses'][k]['choices'][choice_type][choice_idx] += ' |'
                for source in src_d['options'][j]['sources']:
                    data[i]['responses'][k]['choices'][choice_type][choice_idx] += ' ' + source
                j += 1

with open(data_path, 'w', encoding='utf-8') as fout:
    fout.write(json.dumps(data, indent=4))


