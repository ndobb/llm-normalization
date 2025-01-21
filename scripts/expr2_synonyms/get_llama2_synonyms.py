import os
import json
import psycopg2
import requests
import re
from tqdm import tqdm

template = """You are an expert system for providing synonyms of phrases which appear in medical abstracts and clinical notes to aid in normalizing phrases to UMLS concepts. Given a surrounding text context, respond with up to {syn_count} relevant synonyms to the bracketed term in bullet points. 
If there is an anatomical word, type of disorder, or behavior the bracketed term may be referring to, include that reference in your synonyms.
The specific term to find synonyms for will be surrounded by double-brackets, like {{term}}.
"""

models_path = '/Users/ndobb/work/llama.cpp/models'
model_path = os.path.join(models_path, 'vicuna-13b-v1.5.Q4_K_M.gguf')
outpath = f'./data/synonyms/medmentions_vicuna13b_synonyms.json'

regex = re.compile('{{(.*?)}}')

def main():
    concepts = get_concepts()
    data, docs = get_data(concepts, errored_only=False)

    if os.path.exists(outpath):
        with open(outpath, 'r', encoding='utf-8') as fin:
            synonyms = json.loads(fin.read())
    else:
        synonyms = []
    syn_count = 3

    # Filter already processed
    print(f'{len(data)} data entries')
    processed = set([d['id'] for d in synonyms])
    print(f'{len(processed)} already processed')
    data = {k:v for k,v in data.items() if v['id'] not in processed}
    print(f'{len(data)} yet to be processed')

    for i, d in enumerate(tqdm([d for _,d in data.items()])):
        if "is_discontiguous" in d and d["is_discontiguous"]: continue
        context = get_context(d, docs[d['filename']])
        concept_strs = concepts[d['cui']][:3]

        #print(f'Running {i} of {len(data)} using {os.getenv("OPENAI_MODEL_NAME")}...')
        
        existing = [x for x in synonyms if d['filename'] == x['filename'] and d['cui'] == x['cui'] and d['text'] == get_text(x['input'])]
        if any(existing):
            response = existing[0]['response']
        else:
            system_prompt = template.replace('{syn_count}', str(syn_count))
            full_prompt = f'USER: {system_prompt} Please find synonyms given this context: "{context}". ASSISTANT:'

            response = call_llama2(full_prompt)

        synonyms.append({ 
            'id': d['id'], 'filename': d['filename'], 'cui': d['cui'], 'strs': concept_strs, 
            'input': context, 'response': response 
        })

        with open(outpath, 'w+', encoding='utf-8') as fout:
            fout.write(json.dumps(synonyms, indent=4))
    
    with open(outpath, 'w+', encoding='utf-8') as fout:
        fout.write(json.dumps(synonyms, indent=4))


def get_text(context):
    x = re.search(regex, context)
    if x:
        return x.group().replace('{','').replace('}','')
    return ''


def call_llama2(input):
    json_data = {'prompt': input, 'n_predict': 128}
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:8080/completion', headers=headers, json=json_data)
    return json.loads(response.text)['content']


def get_context(d, doc, window_len=300):
    idxs = d['indices']
    if len(idxs) == 2: 
        beg, end = idxs[0], idxs[1]
    else: 
        beg, end = idxs[0][0], idxs[0][1]
    beg, end = beg[0] if type(beg) == list else beg, end[-1] if type(end) == list else end
    text = doc['text']
    context = '...' + text[beg-window_len:beg] + '{{' + d['text'] + '}}' + text[end:end+window_len] + '...'

    return context


def get_data(concepts, errored_only=True):
    docs, annotation_data = {}, {}
    with open(os.path.join('./data', 'llm_normalization_medmentions_test.json'), 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())
        for doc_id, doc in data['data'].items():
            docs[doc_id] = doc
            for ann in doc['annotations']:
                id = f'{doc["filename"]}_{ann["indices"][0]}'
                ann['filename'] = doc['filename']
                annotation_data[id] = ann
            
    with open('./data/error_analysis.json', 'r', encoding='utf-8') as fin:
        error_data = json.loads(fin.read())['data']

    for k,v in annotation_data.items():
        annotation_data[k] = v | error_data[k]

    annotation_data = {k:v for k,v in annotation_data.items() if v['cui'] in concepts}

    if errored_only:
        annotation_data = {k:v for k,v in annotation_data.items() if not v['bm25']['matched']}

    return annotation_data, docs


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
