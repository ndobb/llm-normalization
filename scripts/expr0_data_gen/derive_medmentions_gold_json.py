import os
import json
import psycopg2

data_path = './data/corpus_pubtator.txt'
test_ids_path = './data/corpus_pubtator_pmids_test.txt'


def get_disorder_cuis():
    data = set()
    with get_vector_db_conn() as conn:
        cur = conn.cursor()
        cur.execute('SELECT cui FROM concept_strs')
        for row in cur.fetchall():
            data.add(row[0])
    return data


def get_vector_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="umls",
        user="postgres",
        password=""
    )

disorder_cuis = get_disorder_cuis()

with open(test_ids_path, 'r') as fin:
    test_ids = set(fin.read().split('\n'))

with open(data_path, 'r', encoding='utf-8') as fin:
    data = {'train': {}, 'test': {}}
    id = ''
    for line in fin.readlines():
        if not line.strip():
            continue
        line = line.strip()
        is_meta_line = line[8] == '|'
        if is_meta_line:
            parts = line.split('|')
            id = parts[0]
            split = 'train' if id not in test_ids else 'test'
            if id not in data[split]:
                data[split][id] = { 'filename': id, 'title': '', 'abstract': '', 'annotations': [] }
            if parts[1] == 't':
                data[split][id]['title'] = '|'.join(parts[2:])
            elif parts[1] == 'a':
                data[split][id]['abstract'] = '|'.join(parts[2:])
                data[split][id]['text'] = data[split][id]['title'] + '\n' + data[split][id]['abstract']
        else:
            parts = line.split('\t')
            id, text, stys, cui = parts[0], parts[3], parts[4].split(','), parts[5]
            beg_idx, end_idx = int(parts[1]), int(parts[2])
            
            if cui not in disorder_cuis:
                continue

            annotation = {
                'text': text, 
                'cui': cui,
                'indices': [beg_idx, end_idx],
                'source': data_path,
                'is_abbrevation': len(text) == 1 or all([x in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789-' for x in text])
            }

            data[split][id]['annotations'].append(annotation)

with open('./data/medmentions.json', 'w+', encoding='utf-8') as fout:
    fout.write(json.dumps({'data': data}, indent=4))


    