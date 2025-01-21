import os
import re
import sys
import json
from tqdm import tqdm
import requests
from ollama import chat, ChatResponse
from pgvector.psycopg2 import register_vector
from openai import OpenAI
import random

sys.path.append('/Users/ndobb/work/llm-normalization/')

from scripts.expr1_baseline.get_retrieval_perf_bounds import (
    get_vector_db_conn, get_embedding, get_derived_synonyms,parse_llm_synonyms,
    get_topk_similar_concepts, get_cached_synonym_embedding, get_openai_key
)


openai = OpenAI(api_key=get_openai_key())
data_path = os.path.join('./data', 'medmentions_vicuna13b_prompt_variants_by_source_1000.json')


template = """You are an expert system for determining appropriate UMLS concepts for a given term. Given a surrounding text context and list of possible UMLS concepts, filter the possible UMLS concepts to only those which could possibly be appropriate, at most {max_answers}. 
{flexible_or_no}
{CoT_or_no}
{output_type_template}
The specific term to filter UMLS concepts for will be surrounded by double-brackets, like {{term}}.

Here is the context: 
"{context}"

{concepts}
"""

return_sty  = lambda c: '(' + ', '.join(c['stys']) + ')'
return_cui  = lambda c,i: c['cui'] + ' ' + return_str(c,i) + ' ' + return_sty(c)
return_idx  = lambda c,i: '(' + str(i) + ') ' + return_str(c,i) + ' ' + return_sty(c)
return_idx1 = lambda c,i: '(' + str(i+1) + ') ' + return_str(c,i) + ' ' + return_sty(c)
return_str  = lambda c,i: '"' + c['str'] + ('; ' + c['pt'] if c['pt'] and c['str'] != c['pt'] else '') + '"'

output_types = [
    ['cui', return_cui, 'Respond only in a JSON object of concept IDs like {CoT_or_no}. Strictly output only concept IDs in your <answer>.'],
    ['idx1', return_idx1, 'Respond only in a JSON object of concept indices like {CoT_or_no}. Strictly output only index numbers in your <answer>.']
]

term_regex = re.compile('{{(.*?)}}')


def main():
    conn      = get_vector_db_conn()
    data      = get_data()
    synonyms = get_derived_synonyms()
    preferred_terms = get_preferred_terms()
    register_vector(conn)
    sample_size = 1000

    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as fin:
            prompt_variant_results = json.loads(fin.read())
    else:
        prompt_variant_results = data
        random.seed(42)
        random.shuffle(prompt_variant_results)

    for d in tqdm(prompt_variant_results[:sample_size]):
        context = get_context(d)
        if 'responses' not in d:
            d['responses'] = {}
        for output_type, output_func, output_type_template in output_types:
            options = get_options(conn, d['id'], synonyms.get(d['id']), preferred_terms)
            for use_CoT in [True,False]:
                for max_answers in [2]:
                    out_type = output_type + '_' + str(max_answers)
                    if use_CoT:
                        out_type += '_CoT'
                    out_type += '|embedding'

                    if out_type in d['responses']:
                        continue
                    
                    direct_choices, indirect_choices, concepts_template = get_choices(output_type, output_func, options)
                    correct = get_gold(output_type, d, options)
                    output_template = output_type_template.replace('{CoT_or_no}', '{"answer": <concept IDs>, "reason": <reason>}' 
                                                                if use_CoT else '{"answer": <concept IDs>}')

                    system_prompt = template \
                        .replace('{output_type_template}', output_template) \
                        .replace('{context}', context) \
                        .replace('{concepts}', concepts_template) \
                        .replace('{CoT_or_no}', "Reason step-by-step about why concepts are appropriate." if use_CoT else "") \
                        .replace('{flexible_or_no}', "Be flexible and generous in your interpretation, and general concepts as well as specific appropriate concepts should be included.") \
                        .replace('{max_answers}', str(max_answers))
                    full_prompt = system_prompt #f'USER: {system_prompt}\. ASSISTANT: '

                    if not any(options):
                        response = ''
                    elif len(options) == 1:
                        response = (direct_choices + indirect_choices)[0]
                    else:
                        response = call_vicuna(full_prompt)

                        d['responses'][out_type] = {
                            'correct': correct, 'input': full_prompt, 'response': response, 
                            'choices': {'direct': direct_choices, 'indirect': indirect_choices}
                        }
        with open(data_path, 'w', encoding='utf-8') as fout:        
            fout.write(json.dumps(prompt_variant_results, indent=4))


def get_options(conn, id, synonyms, preferred_terms):
    synonyms = synonyms['response'] if synonyms else ''
    
    # Get direct matches
    with get_vector_db_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT id, embedding FROM annotation WHERE id = %s', (id,))
            row = cursor.fetchone()
            if row:
                embedding = row[1]
                direct = get_topk_similar_concepts(conn, 5, embedding)
            else:
                direct = []

    # Get synonym matches
    indirect = []
    for synonym in parse_llm_synonyms(synonyms):
        if synonym:
            cached = get_cached_synonym_embedding(conn, synonym)
            if cached:
                found_syns = cached['found']
            else:
                embedding = get_embedding(openai, synonym).data[0].embedding
                found_syns = get_topk_similar_concepts(conn, 1, embedding)
            if any(found_syns):
                found_syns[0]['synonym'] = synonym
                indirect.append(found_syns[0])

    output = []
    for o in direct:
        if any([x for x in output if o['cui'] == x['cui']]):
            continue
        o['isDirect'] = True
        o['sources'] = ['embedding']
        o['synonym'] = None
        o['str'] = o['text']
        o['pt'] = preferred_terms[o['cui']]['str'] if o['cui'] in preferred_terms else o['text']
        del o['text']
        output.append(o)
    for o in indirect:
        if any([x for x in output if o['cui'] == x['cui']]):
            continue
        o['isDirect'] = False
        o['sources'] = ['embedding']
        o['str'] = o['text']
        o['stys'] = preferred_terms[o['cui']]['sty'] if o['cui'] in preferred_terms else []
        o['pt'] = preferred_terms[o['cui']]['str'] if o['cui'] in preferred_terms else o['text']
        del o['text']
        output.append(o)
    
    return output


def get_choices(output_type, output_func, options):
    direct_choices = [output_func(x,i) for i,x in enumerate(options) if x['isDirect']]
    indirect_choices = [output_func(x,i) for i,x in enumerate(options) if not x['isDirect']]
    if output_type == 'str':
        direct_choices = list(set(direct_choices))
        indirect_choices = list(set(indirect_choices))

    if not any(direct_choices):
        direct_choices = indirect_choices
        indirect_choices = []
    
    concepts_template = 'Here are possible concepts:\n\n' + '\n'.join(direct_choices)
    if any(indirect_choices):
        concepts_template += '\n\nThe following are also possible but less likely:\n\n' + '\n'.join(indirect_choices)

    return direct_choices, indirect_choices, concepts_template


def get_derived_synonyms():
    synonym_path = os.path.join('./data', 'synonyms', 'medmentions_vicuna13b_synonyms.json')

    output = {}
    with open(synonym_path, 'r', encoding='utf-8') as fin:
        data = json.loads(fin.read())

    for d in data:
        output[d['id']] = d
    
    return output


def get_gold(output_type, d, choices):
    match = [(i,c) for i,c in enumerate(choices) if c['cui'] == d['cui']]
    if output_type == 'cui':
        if any(match): return d['cui']
    elif output_type == 'idx':
        if any(match): return str(match[0][0])
    elif output_type == 'idx1':
        if any(match): return str(match[0][0]+1)
    elif output_type == 'str':
        if any(match): return str(match[0][1]['str'])
    
    return None


def call_vicuna(input):
    response = chat(model='vicuna:13b', messages=[
        {
            'role': 'user',
            'content': input,
        },
    ])
    return response['message']['content']


def call_llama2(input, n_predict=128, temperature=0.2):
    json_data = {'prompt': input, 'n_predict': n_predict, 'temperature': temperature}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post('http://localhost:8080/completion', headers=headers, json=json_data)
    except Exception as ex:
        return 'Error'
    return json.loads(response.text)['content']


def get_context(d, window_len=300):
    idxs = d['indices']
    if len(idxs) == 2: 
        beg, end = idxs[0], idxs[1]
    else: 
        beg, end = idxs[0][0], idxs[0][1]
    beg, end = beg[0] if type(beg) == list else beg, end[-1] if type(end) == list else end
    text = d['doc_text']
    context = '...' + text[beg-window_len:beg] + '{{' + d['text'] + '}}' + text[end:end+window_len] + '...'

    return context


def get_data():
    with open(data_path, 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())
        return data
    

def get_base_options():
    output = {}
    with open(os.path.join('./data', 'prompt_variant_test_vicuna_concepts.json'), 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())

    for d in data:
        output[d['id']] = d

    return output


def get_preferred_terms():
    output = {}
    with open(os.path.join('./data', 'umls_preferred_terms.json'), 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())

    for d in data['data']:
        output[d['cui']] = d

    return output

main()