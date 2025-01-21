import os
import re
import sys
import json
import requests
from openai import OpenAI
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

sys.path.append('/Users/ndobb/work/llm-normalization/')

from scripts.expr1_baseline.get_retrieval_perf_bounds import (
    get_vector_db_conn, get_embedding, get_derived_synonyms,parse_llm_synonyms,
    get_topk_similar_concepts, get_cached_synonym_embedding
)

data_path = os.path.join('./data', 'medmentions_gpt3.5_prompt_variants_yesno.json')


template = """You are an expert system for determining appropriate UMLS concepts for a given term. Given a surrounding text context and a term, determine whether a given UMLS concept means the same thing as {term}. 

Here is the context: 
"{context}"

Is the concept "{concept}" the same as "{term}" in this context? Answer true if yes, false if no. {CoT_or_no}
"""

return_sty  = lambda c: '(' + ', '.join(c['stys']) + ')'
return_cui  = lambda c,i: c['cui'] + ' ' + return_str(c,i) + ' ' + return_sty(c)
return_str  = lambda c,i: c['str'] + ('; ' + c['pt'] if c['pt'] and c['str'] != c['pt'] else '')

term_regex = re.compile('{{(.*?)}}')

def get_openai_key():
    key = ''
    with open('./.env', 'r') as fin:
        for line in fin.readlines():
            if line.startswith('OPENAI_API_KEY'):
                key = line.replace('OPENAI_API_KEY=','').strip()
    return key
openai = OpenAI(api_key=get_openai_key())


def main():
    model = 'gpt-3.5-turbo-0125'
    prompt = PromptTemplate(template=template, input_variables=["term", "context", "concept", "CoT_or_no"])
    llm = ChatOpenAI(model_name=model, temperature=0.2)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    conn      = get_vector_db_conn()
    data      = get_data()
    synonyms = get_derived_synonyms()
    preferred_terms = get_preferred_terms()
    register_vector(conn)

    for i, d in enumerate(data):
        print(f'Document {i}')
        modified = False
        context = get_context(d)
        if 'responses' in d:
            print(f' Get options')
            options = get_options(conn, d['id'], synonyms.get(d['id']), preferred_terms)
            for use_CoT in [True,False]:
                out_type = 'yesno'
                if use_CoT:
                    out_type += '_CoT'

                if out_type in d['responses']:
                    for response in d['responses'][out_type]['responses']:
                        if any([x for x in options if x['cui'] == response['answer']]):
                            response['sources'].append('embedding')
                            modified = True
                        response['sources'] = list(set(response['sources']))

                for j, o in enumerate(options):
                    if out_type in d['responses'] and not any([x for x in d['responses'][out_type]['responses'] if x['answer'] == o['cui']]):
                        text = return_str(o, 0) + ' ' + return_sty(o)
                        Cot = 'Reason step-by-step briefly about why a concept is appropriate. Structure your output in JSON as { "answer": <true|false>, "reason": <reason> } ' \
                            if use_CoT else 'Structure your output in JSON as { "answer": <true|false> })'
                        
                        print(f'Document {i} option {j} calling GPT...')

                        response = llm_chain.run(term=d["text"], context=context, concept=text, CoT_or_no=Cot)
                        d['responses'][out_type]['responses'].append({ 
                            'answer': o['cui'], 'input': text, 'response': response, 'isDirect': o['isDirect'], 
                            'sources': list(set(o['sources'])), 'synonym': o['synonym'] 
                        })
                        modified = True

        if modified and i % 50 == 0:
            print(f'  Saving data')
            with open(data_path, 'w+', encoding='utf-8') as fout:
                fout.write(json.dumps(data, indent=4))

    print(f'  Saving data')
    with open(data_path, 'w+', encoding='utf-8') as fout:
        fout.write(json.dumps(data, indent=4))


def get_options(conn, id, synonyms, preferred_terms):
    synonyms = synonyms['response'] if synonyms else ''
    
    # Get direct matches
    print(f' K similar')
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
    print(f' Synonyms')
    indirect = []
    for synonym in parse_llm_synonyms(synonyms):
        if synonym:
            print(f' Synonym cache')
            cached = get_cached_synonym_embedding(conn, synonym)
            if cached:
                found_syns = cached['found']
            else:
                print(f' Get embedding')
                embedding = get_embedding(openai, synonym).data[0].embedding
                print(f' Get K similiar synonym')
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