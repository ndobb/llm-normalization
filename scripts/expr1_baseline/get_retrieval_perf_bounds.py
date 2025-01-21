import os
import json
import psycopg2
import numpy as np
import requests
from pgvector.psycopg2 import register_vector
from tqdm import tqdm

from quickumls import QuickUMLS
from retriv import SparseRetriever
from openai import OpenAI



def main():
    conn     = get_vector_db_conn()
    gold     = get_gold_annotations()
    concepts = get_umls_concepts(conn)
    unq_cuis = set([c['cui'] for c in concepts])
    synonym_datasets = get_derived_synonyms()
    register_vector(conn)

    #bm25 = SparseRetriever().index(concepts)
    #quickUmls = QuickUMLS('./data/QuickUMLS')
    openai = OpenAI(api_key=get_openai_key())

    extractors = [
        #[ 'metamap',   lambda text, embedding, k: normalize_by_metamap(text), True ],
        #[ 'quickumls', lambda text, embedding, k: normalize_by_quickUmls(quickUmls, text), True ],
        #[ 'bm25',      lambda text, embedding, k: get_bm25_unique_cuis(bm25, text, k), False ],
        [ 'embedding',    lambda text, embedding, k: get_topk_similar_concepts(conn, k, embedding), False ]
    ]

    output = {}
    start_k, max_k, increment = 5, 5, 5
    k = start_k

    print(f'Total gold entries: {len(gold)}')

    results_to_print = []
    results_json = []

    for use_synonyms in [True]:
        for synonym_dataset, synonyms in synonym_datasets.items():
            if 'gpt' in synonym_dataset:
                continue
            for extractor_name, extract, run_once_only in extractors:
                orig_extractor_name = extractor_name
                if use_synonyms:
                    extractor_name += '_' + synonym_dataset
                while k <= max_k:
                    if k not in output: 
                        output[k] = {}
                    for syn_k in [1]:
                        for g in tqdm(gold):
                            try:
                                if g['cui'] not in unq_cuis:
                                    continue
                                out = g
                                found = extract(g['text'], g['embedding'], k)

                                if use_synonyms:
                                    synonym_entry = synonyms.get(out['id'])
                                    if synonym_entry:
                                        for synonym in parse_llm_synonyms(synonym_entry['response']):
                                            if synonym:
                                                if 'embedding' in extractor_name:
                                                    cached = get_cached_synonym_embedding(conn, synonym)
                                                    if cached:
                                                        embedding = cached['embedding']
                                                        found_syns = cached['found']
                                                    else:
                                                        embedding = get_embedding(openai, synonym).data[0].embedding
                                                        found_syns = extract(synonym, embedding, syn_k)
                                                        if any(found_syns):
                                                            cui_id, cui, cui_text = found_syns[0]['id'], found_syns[0]['cui'], found_syns[0]['text']
                                                        else:
                                                            cui_id, cui, cui_text = None, None, None
                                                        save_cached_synonym_embeddings(conn, synonym, embedding, cui_id, cui, cui_text)

                                                    found = merge_unique_cuis(found, found_syns)
                                                    continue
                                                else:
                                                    embedding = None # Embeddings are not used by other methods, so nulling is benign

                                                found_syns = extract(synonym, embedding, syn_k)
                                                found = merge_unique_cuis(found, found_syns)
                                else:
                                    syn_k = None

                                cuis = set([x['cui'] for x in found if x['cui'] in unq_cuis])
                                out[extractor_name] = { 'found': found, 'matched': g['cui'] in cuis }
                                
                                if out['id'] not in output[k]:
                                    output[k][out['id']] = out
                                else:
                                    output[k][out['id']][extractor_name] = out[extractor_name]
                            except Exception as ex:
                                print(f'Error! {ex}')

                        # Recall
                        r_num = len([x for _, x in output[k].items() if x[extractor_name]["matched"]])
                        r_denom = len(output[k])
                        r = round(r_num * 1.0 / r_denom * 100, 2)

                        # Precision
                        p_num = len([x for _, x in output[k].items() if x[extractor_name]["matched"]])
                        p_denom = sum([len(x[1][extractor_name]['found']) for x in output[k].items()])
                        p = round(p_num * 1.0 / p_denom * 100, 2)

                        # F1
                        f1 = round(2 * (p * r) / (p + r), 2)

                        results = { 
                            'k': None if run_once_only else k, 'syn_k': None if not use_synonyms else syn_k, 
                            'extractor': orig_extractor_name, 'dataset': synonym_dataset,
                            'synonyms': use_synonyms, 'precision': p, 'recall': r, 'f1': f1
                        }
                        to_print = f'k={k} syn_k={syn_k} {extractor_name}{"_syns" if use_synonyms else ""} - R: {r}, P: {p}, F1: {f1}'
                        print(to_print)
                        results_to_print.append(to_print)
                        results_json.append(results)

                        #with open('./data/experiments_1_2_results.json', 'w+', encoding='utf-8') as fout:
                        #    fout.write(json.dumps({ 'data': results_json }, indent=4))

                        if not use_synonyms:
                            break

                    k += increment

                    if run_once_only:
                        break 
                k = start_k
            if not use_synonyms:
                break
    
    print()
    for results in sorted(results_to_print):
        print(results)

    #for k, _ in output.items():
    #   for id, val in output[k].items():
    #       del val['embedding']

    #with open('./data/error_analysis.json', 'w+', encoding='utf-8') as fout:
    #   fout.write(json.dumps({ 'data': output[k] }, indent=4))


def get_cached_synonym_embedding(conn, synonym):
    sql = 'SELECT embedding, cui_id, cui, cui_text from synonym_embeddings WHERE synonym = %s LIMIT 1'
    with conn.cursor() as cur:
        cur.execute(sql, (synonym,))
        for row in cur.fetchall():
            return { 'embedding': row[0], 'found': [{'id': row[1], 'cui': row[2], 'text': row[3]}] }
    return None


def save_cached_synonym_embeddings(conn, synonym, embedding, cui_id, cui, cui_text):
    if len(synonym) > 200:
        return
    try:
        sql = 'INSERT INTO synonym_embeddings (synonym, embedding, cui_id, cui, cui_text) SELECT %s, %s, %s, %s, %s'
        with conn.cursor() as cur:
            cur.execute(sql, (synonym, embedding, cui_id, cui, cui_text))
    except Exception as ex:
        print(ex)
    conn.commit()


def merge_unique_cuis(orig, added):
    output = orig
    orig_cuis = set([c['cui'] for c in orig])
    for c in added:
        if c['cui'] not in orig_cuis:
            output.append(c)

    return output


def parse_llm_synonyms(text):
    output = []
    if ':' in text:
        text = text[text.find(':'):]
    for line in text.split('\n'):
        line = line.replace('*', '')
        if line.startswith('-'):
            line = line[1:]
        output.append(line.strip())
    
    return output


def get_derived_synonyms():
    synonym_datasets = {}
    synonym_path = os.path.join('./data', 'synonyms')

    for file in os.listdir(synonym_path):
        alias = 'gpt3.5' if 'gpt' in file else 'vicuna'
        with open(os.path.join(synonym_path, file), 'r', encoding='utf-8') as fin:
            if 'embeddings' in file:
                continue
            data = json.loads(fin.read())
            synonym_datasets[alias] = {v['id']: v for v in data}

    with open(os.path.join('./data', 'medmentions.json'), 'r', encoding='utf-8') as fin:
        medmentions = json.loads(fin.read())
        synonym_datasets['vicuna'] = {k:v for k,v in synonym_datasets['vicuna'].items() if v['filename'] in medmentions['data']['test']}
    
    return synonym_datasets


def get_bm25_unique_cuis(index, text, k):
    cuis = {}
    hits = index.search(text, cutoff=k*5, return_docs=True)

    for hit in hits:
        if hit['cui'] in cuis:
            continue
        cuis[hit['cui']] = hit
        if len(cuis) == k:
            break

    return [v for _,v in cuis.items()]


def normalize_by_quickUmls(quickUmls, text):
    found = quickUmls.match(text, best_match=False)
    if not any(found):
        return []
    return found[0]


def normalize_by_metamap(text):
    response = requests.get(f'http://localhost:5001/api/nlp/normalize?text={text}')
    if response.ok:
        data = json.loads(response.text)
        return data
    return []


def get_vector_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="vectorexample",
        user="postgres",
        password=""
    )


def get_umls_concepts(conn):
    sql = 'SELECT id, cui, str, sty FROM umls'
    cur = conn.cursor()
    cur.execute(sql)
    concepts = []
    while True:
        row = cur.fetchone()
        if not row:
            break
        id, cui, name, stys = row[0], row[1], row[2], json.loads(row[3])
        concept = { 'id': id, 'cui': cui, 'text': name, 'stys': stys}
        concepts.append(concept)

    return concepts


def get_topk_similar_concepts(conn, k, query_embedding):
    cur = conn.cursor()
    cur.execute(f"SELECT id, cui, str, sty FROM umls ORDER BY embedding <=> %s LIMIT {k}", (np.array(query_embedding),))
    topk = []
    for row in cur.fetchall():
        id, cui, name, stys = row[0], row[1], row[2], json.loads(row[3])
        rec = { 'id': id, 'cui': cui, 'text': name, 'stys': stys }
        topk.append(rec)

    return topk


def get_gold_annotations():
    with open(os.path.join('./data', 'medmentions.json'), 'r', encoding='utf-8') as fin:
        medmentions = json.loads(fin.read())

    embeddings = {}
    with get_vector_db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id, embedding FROM annotation')
            while True:
                row = cursor.fetchone()
                if not row:
                    break
                id, embedding = row[0], json.loads(row[1])
                embeddings[id] = embedding

    annotations = []
    for _, v in medmentions['data']['test'].items():
        for ann in v['annotations']:
            ann['id']        = v['filename'] + '_' + str(ann['indices'][0])
            ann['doc_text']  = v['text']
            ann['filename']  = v['filename']
            ann['embedding'] = embeddings.get(ann['id'])
            annotations.append(ann)

    return annotations


def get_embedding(client, text, model='text-embedding-3-small'):
    return client.embeddings.create(input = [text], model=model)


def get_openai_key():
    key = ''
    with open('./.env', 'r') as fin:
        for line in fin.readlines():
            if line.startswith('OPENAI_API_KEY'):
                key = line.replace('OPENAI_API_KEY=','').strip()
    return key


if __name__ == '__main__':
    main()    