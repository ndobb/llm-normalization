import os
import json
import requests
import random
import re
from tqdm import tqdm

random.seed(42)

template = """You are an expert system for determining appropriate UMLS concepts for a given term. Given a surrounding text context and a term,  determine whether a given UMLS concept means the same thing as {{term}}. 

Here is the context: 
"{context}"

Is the concept "{concept}" the same as {{term}} in this context? Answer true is yes, false if no. {CoT_or_no}
"""

return_sty  = lambda c: '(' + ', '.join(c['stys']) + ')'
return_cui  = lambda c,i: c['cui'] + ' ' + return_str(c,i) + ' ' + return_sty(c)
return_str  = lambda c,i: c['str'] + ('; ' + c['pt'] if c['pt'] and c['str'] != c['pt'] else '')

outpath = f'./data/medmentions_vicuna13b_prompt_variants_yesno.json'
term_regex = re.compile('{{(.*?)}}')

def main():
    data = get_data()

    if os.path.exists(outpath):
        with open(outpath, 'r', encoding='utf-8') as fin:
            prompt_variant_results = json.loads(fin.read())
    else:
        prompt_variant_results = data
        random.seed(42)
        random.shuffle(prompt_variant_results)  

    # Filter already processed
    print(f'{len(prompt_variant_results)} data entries')
    print(f'{len([d for d in prompt_variant_results if "options" in d])} yet to be processed')

    for d in tqdm(prompt_variant_results):
        if 'options' not in d:
            continue
        context = get_context(d)
        options = d['options']
        d['responses'] = {}
        for use_CoT in [True]:
            out_type = 'yesno'
            if use_CoT:
                out_type += '_CoT'
            cui = d['cui']

            system_prompt = template \
                .replace('{context}', context) \
                .replace('{CoT_or_no}', \
                         'Reason step-by-step about why a concept is appropriate. Structure your output in JSON as { "answer": <true|false>, "reason": <reason> } ' if use_CoT else 'Structure your output in JSON as { "answer": <true|false> })')
            
            responses = []

            for i, concept in enumerate(options):
                text = return_str(concept, 0) + ' ' + return_sty(concept)
                full_prompt = f'USER: {system_prompt.replace("{concept}", text).replace("{{term}}", "{{" + d["text"] + "}}")}\n. ASSISTANT: '

                response = call_llama2(full_prompt)
                responses.append({ 
                    'answer': concept['cui'], 'input': text, 'response': response, 'isDirect': concept['isDirect'], 
                    'sources': concept['sources'], 'synonym': concept['synonym'] 
                })

                d['responses'][out_type] = {
                    'answer': cui, 'input': full_prompt, 'responses': responses
                }

                with open(outpath, 'w+', encoding='utf-8') as fout:
                    fout.write(json.dumps(prompt_variant_results, indent=4))

        del d['options']


def call_llama2(input, n_predict=128, temperature=0.2):
    json_data = {'prompt': input, 'n_predict': n_predict, 'temperature': temperature}
    headers = {'Content-Type': 'application/json'}
    response = requests.post('http://localhost:8080/completion', headers=headers, json=json_data)
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
