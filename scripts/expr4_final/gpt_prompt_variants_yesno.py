import os
import json
import random
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from tqdm import tqdm

load_dotenv()
random.seed(42)

template = """You are an expert system for determining appropriate UMLS concepts for a given term. Given a surrounding text context and a term, determine whether a given UMLS concept means the same thing as {term}. 

Here is the context: 
"{context}"

Is the concept "{concept}" the same as "{term}" in this context? Answer true if yes, false if no. {CoT_or_no}
"""

return_sty  = lambda c: '(' + ', '.join(c['stys']) + ')'
return_cui  = lambda c,i: c['cui'] + ' ' + return_str(c,i) + ' ' + return_sty(c)
return_str  = lambda c,i: c['str'] + ('; ' + c['pt'] if c['pt'] and c['str'] != c['pt'] else '')

outpath = f'./data/medmentions_gpt3.5_prompt_variants_yesno.json'


def main():
    model = 'gpt-3.5-turbo-0125'
    print(model)

    prompt = PromptTemplate(template=template, input_variables=["term", "context", "concept", "CoT_or_no"])
    llm = ChatOpenAI(model_name=model, temperature=0.2)
    llm_chain = LLMChain(prompt=prompt, llm=llm)
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
            
            responses = []

            for i, concept in enumerate(options):
                text = return_str(concept, 0) + ' ' + return_sty(concept)
                Cot = 'Reason step-by-step briefly about why a concept is appropriate. Structure your output in JSON as { "answer": <true|false>, "reason": <reason> } ' \
                    if use_CoT else 'Structure your output in JSON as { "answer": <true|false> })'

                # input_variables=["term", "context", "concept", "CoT_or_no"])
                response = llm_chain.run(term=d["text"], context=context, concept=text, CoT_or_no=Cot)
                responses.append({ 
                    'answer': concept['cui'], 'input': text, 'response': response, 'isDirect': concept['isDirect'], 
                    'sources': concept['sources'], 'synonym': concept['synonym'] 
                })

                d['responses'][out_type] = {
                    'answer': cui, 'responses': responses, 'input': template.replace('{term}', d["text"]).replace('{context}', context).replace('{concept}', text).replace('{CoT_or_no}', Cot), 
                }

        with open(outpath, 'w+', encoding='utf-8') as fout:
            fout.write(json.dumps(prompt_variant_results, indent=4))

        del d['options']


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
    with open(os.path.join('./data', 'prompt_variant_test_gpt3.5_concepts.json'), 'r', encoding='utf-8') as fin:        
        data = json.loads(fin.read())
        return data


main()
