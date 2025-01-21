import os
import json
import requests
import random
import re
from tqdm import tqdm

random.seed(42)

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
    #['idx', return_idx,   'Output your results in a JSON object of concept indices like {"answer": <indices>, "reason": <reason>}. Strictly output only index numbers in your <answer>.'],
    ['idx1', return_idx1, 'Respond only in a JSON object of concept indices like {CoT_or_no}. Strictly output only index numbers in your <answer>.'],
    #['str', return_str,   'Output your results in a JSON object of concept descriptions like ```json{"answer": <descriptions>, "reason": <reason>}```. Strictly output only concept descriptions in your <answer>.']
]

outpath = f'./data/medmentions_vicuna13b_prompt_variants.json'
term_regex = re.compile('{{(.*?)}}')

def main():
    data = get_data()
    sample_size = 1000

    # 1st option is correct in 58.3%
    # 2nd option is correct in 14.4%
    # 3rd option is correct in 4.3%

    if os.path.exists(outpath):
        with open(outpath, 'r', encoding='utf-8') as fin:
            prompt_variant_results = json.loads(fin.read())
    else:
        prompt_variant_results = data

    # Filter already processed
    print(f'{len(prompt_variant_results)} data entries')
    print(f'{len([d for d in prompt_variant_results if "options" in d])} yet to be processed')

    random.seed(42)
    random.shuffle(prompt_variant_results)

    for i, d in enumerate(tqdm(prompt_variant_results[:sample_size])):
        if 'options' not in d:
            continue
        context = get_context(d)
        if 'responses' not in d:
            d['responses'] = {}
        for output_type, output_func, output_type_template in output_types:
            options = d['options']
            for source_limit in ['all']:
                if source_limit != 'all':
                    options = [o for o in options if source_limit in o['sources']]
                for use_CoT in [True, False]:
                    for max_answers in [2]:
                        for be_flexible in [True]:
                            out_type = output_type + '_' + str(max_answers)
                            if use_CoT:
                                out_type += '_CoT'
                            if source_limit != 'all':
                                out_type += '_' + source_limit

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
                                .replace('{flexible_or_no}', "Be flexible and generous in your interpretation, and general concepts as well as specific appropriate concepts should be included." if be_flexible else "") \
                                .replace('{max_answers}', str(max_answers))
                            full_prompt = f'USER: {system_prompt}\. ASSISTANT: '

                            response = call_llama2(full_prompt)
                            d['responses'][out_type] = {
                                'correct': correct, 'input': full_prompt, 'response': response, 
                                'choices': {'direct': direct_choices, 'indirect': indirect_choices}
                            }

                            with open(outpath, 'w+', encoding='utf-8') as fout:
                                fout.write(json.dumps(prompt_variant_results, indent=4))
        del d['options']


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


def call_llama2(input, n_predict=128, temperature=0.2):
    json_data = {'prompt': input, 'n_predict': n_predict, 'temperature': temperature}
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post('http://localhost:8080/completion', headers=headers, json=json_data)
    except:
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
    with open(os.path.join('./data', 'prompt_variant_test_vicuna_concepts.json'), 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())
        return data


main()
