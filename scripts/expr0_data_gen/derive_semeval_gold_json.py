import os
import json

base_path        = './data/SemEval_datasets/shareclef-ehealth-evaluation-lab-2014-task-2-disorder-attributes-in-clinical-reports-1.0/ShAReCLEFeHealth2014Task2_test_default_values_with_corpus/'
texts_path       = os.path.join(base_path, 'ShAReCLEFeHealth2104Task2_test_data_corpus/')
values_path      = os.path.join(base_path, 'ShAReCLEFeHealth2014Task2_test_data_default_values')
more_values_path = './data/SemEval_datasets/cuiless2016-1.0.0/CUILESS_DEV'

data = {}
for text_file in os.listdir(texts_path):
    filename = text_file.split('.')[0]
    text_file_path = os.path.join(texts_path, text_file)
    value_paths    = [
        os.path.join(values_path, filename + '.pipe.txt'),
        os.path.join(more_values_path, '-'.join(filename.split('-')[:2]) + '.pipe')
    ]

    with open(text_file_path, 'r', encoding='utf-8') as fin:
        text = fin.read()
        data[filename] = {'filename':filename, 'text': text, 'annotations': []}
    for value_path in value_paths:
        if os.path.exists(value_path):
            with open(value_path, 'r', encoding='utf-8') as fin:
                for line in fin.readlines():
                    if not line:
                        continue
                    parts = line.split('|')
                    indices, cui = parts[1], parts[2]
                    if cui == 'CUI-less':
                        continue
                    index_pairs, indices = indices.split(','), []
                    annotated_text = ''
                    for index_pair in index_pairs:
                        beg, end = [int(x) for x in index_pair.split('-')]
                        annotated_text += ' ' + text[beg:end]
                        indices.append([beg, end])
                    annotated_text = annotated_text.strip()
                    data[filename]['annotations'].append({
                        'text': annotated_text, 
                        'cui': cui,
                        'indices': indices,
                        'source': value_path,
                        'is_abbrevation': len(annotated_text) == 1 or all([x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-' for x in annotated_text]),
                        'is_original': 'CUILESS' not in value_path,
                        'is_discontiguous': len(indices) > 1
                    })
        else:
            print(f'Path "{value_path}" does not exist!')

with open('./data/llm_normalization_semeval2014_test.json', 'w+', encoding='utf-8') as fout:
    fout.write(json.dumps({'data': data}, indent=4))


    
    