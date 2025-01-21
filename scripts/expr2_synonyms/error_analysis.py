import os
import json
import random
import psycopg2

data_paths = [
    ['semeval', os.path.join('./data', 'llm_normalization_semeval2014_test.json')],
    ['medmentions', os.path.join('./data', 'llm_normalization_medmentions_test.json')],
]

def main():
    concepts = get_concepts()
    annotation_data = {}
    docs = {}
    for dataset_name, data_path in data_paths:
        with open(data_path, 'r', encoding='utf-8') as fin:        
            data = json.loads(fin.read())
            for doc_id, doc in data['data'].items():
                docs[doc_id] = doc
                for ann in doc['annotations']:
                    id = f'{doc["filename"]}_{ann["indices"][0]}'
                    ann['filename'] = doc['filename']
                    annotation_data[id] = ann
            
    with open('./data/error_analysis.json', 'r', encoding='utf-8') as fin:
        data = json.loads(fin.read())['data']

    for k,v in data.items():
        data[k] = v | annotation_data[k]

    cuis_not_in_db = set([x['cui'] for k, x in data.items() if x['cui'] not in concepts])
    data = {k:v for k,v in data.items() if v['cui'] in concepts}
    
    correct_if_either = {k:x for k, x in data.items() if x['bm25']['matched'] or x['ada002']['matched']}
    bm25_missed = {k:x for k, x in data.items() if x['bm25']['matched'] == False}
    ada2_missed = {k:x for k, x in data.items() if x['ada002']['matched'] == False}
    overlap_missed = {k:x for k,x in bm25_missed.items() if k in ada2_missed}

    bm25_only_missed = {k:x for k,x in bm25_missed.items() if k not in ada2_missed}
    ada2_only_missed = {k:x for k,x in ada2_missed.items() if k not in bm25_missed}

    all_missed_acronyms = {k:x for k,x in overlap_missed.items() if x['is_abbrevation']}
    missed_not_acronyms = {k:x for k,x in overlap_missed.items() if not x['is_abbrevation']}

    print()
    num_examples = 20
    for dataset in [all_missed_acronyms, missed_not_acronyms]:
        examples = [v for _,v in dataset.items()]
        random.shuffle(examples)
        examples = examples[:num_examples]

        for v in examples:
            if "is_discontiguous" in v:
                if v["is_discontiguous"]:
                    continue
            idxs = v['indices']
            if len(idxs) == 2:
                beg, end = idxs[0], idxs[1]
            else:
                beg, end = idxs[0][0], idxs[0][1]
            beg, end = beg[0] if type(beg) == list else beg, end[-1] if type(end) == list else end

            concept = concepts.get(v['cui'])
            context_window_len = 500
            text = docs[v['filename']]['text']
            context = '...' + text[beg-context_window_len:beg] + '{' + v['text'] + '}' + text[end:end+context_window_len] + '...'
            print(f'"{v["text"]}" - {v["cui"]} - {concept[:3]}\n')
            print(context)
            print('\n\n')


def get_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="umls",
        user="postgres",
        password=""
    )


def get_concepts():
    data = {}
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute('SELECT cui, str FROM concept_strs')
        for row in cur.fetchall():
            cui, text = row[0], row[1]
            if cui in data:
                data[cui].append(text)
            else:
                data[cui] = [text]
    return data

main()
    

