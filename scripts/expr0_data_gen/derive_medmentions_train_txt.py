import os
import json

data_path = './data/corpus_pubtator.txt'
test_ids_path = './data/corpus_pubtator_pmids_test.txt'


template = \
'''<s>USER: {prompt}

ASSISTANT: {response}'''

input_template = """You are an expert system for determining appropriate UMLS concepts for a given term. Given a surrounding text context and list of possible UMLS concepts, filter the possible UMLS concepts to only those which could possibly be appropriate, at most {max_answers}. 
{flexible_or_no}
{CoT_or_no}
{output_type_template}
The specific term to filter UMLS concepts for will be surrounded by double-brackets, like {{term}}.

Here is the context: 
"{context}"

{concepts}"""

output_type_template = 'Respond only in a JSON object of concept IDs.. Strictly output only concept IDs in your answer.'

return_sty  = lambda c: '(' + ', '.join(c['stys']) + ')'
return_cui  = lambda c,i: c['cui'] + ' ' + return_str(c,i) + ' ' + return_sty(c)
return_idx  = lambda c,i: '(' + str(i) + ') ' + return_str(c,i) + ' ' + return_sty(c)
return_idx1 = lambda c,i: '(' + str(i+1) + ') ' + return_str(c,i) + ' ' + return_sty(c)
return_str  = lambda c,i: '"' + c['str'] + ('; ' + c['pt'] if c['pt'] and c['str'] != c['pt'] else '') + '"'

output_types = [
    ['cui', return_cui, 'Respond only in a JSON object of concept IDs like { "answer": [<answers>]}. Strictly output only concept IDs in your answer.'],
    ['idx1', return_idx1, 'Respond only in a JSON object of concept indices like { "answer": [<answers>]}. Strictly output only index numbers in your answer.']
]


def main():
    train = get_train_data()
    with open(os.path.join('./data', 'medmentions_test_finetune.txt'), 'a+', encoding='utf-8') as fout:
        for d in train:

            if d['cui'] not in set([x['cui'] for x in d['options']]):
                x=1

            context = get_context(d)
            _, _, concepts_template = get_choices(return_cui, d['options'])
            response = '{ "answer": ["' + d['cui'] + '"] }\n\n'

            system_prompt = input_template \
                .replace('{output_type_template}', output_type_template) \
                .replace('{context}', context) \
                .replace('{concepts}', concepts_template) \
                .replace('{flexible_or_no}', "Be flexible and generous in your interpretation, and general concepts as well as specific appropriate concepts should be included.") \
                .replace('{max_answers}', str(2))
            full_prompt = template \
                .replace('{prompt}', system_prompt) \
                .replace('{response}', response)
            
            fout.write(full_prompt)


def get_choices(output_func, options):
    direct_choices = [output_func(x,i) for i,x in enumerate(options) if x['isDirect']]
    indirect_choices = [output_func(x,i) for i,x in enumerate(options) if not x['isDirect']]

    if not any(direct_choices):
        direct_choices = indirect_choices
        indirect_choices = []
    
    concepts_template = 'Here are possible concepts:\n\n' + '\n'.join(direct_choices)
    if any(indirect_choices):
        concepts_template += '\n\nThe following are also possible but less likely:\n\n' + '\n'.join(indirect_choices)

    return direct_choices, indirect_choices, concepts_template


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


def get_train_data():
    with open(os.path.join('./data', 'prompt_variant_train_vicuna_concepts.json'), 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())
    return data


main()


    