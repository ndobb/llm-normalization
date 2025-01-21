import os
import json
import psycopg2
import random
import re
import tiktoken
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from tqdm import tqdm

load_dotenv()

template = """
You are an expert system for providing synonyms of phrases which appear in medical abstracts and clinical notes to aid in normalizing phrases to UMLS concepts. Given a surrounding text context, respond with up to {syn_count} relevant synonyms to the bracketed term in bullet points. 
If there is an anatomical word, type of disorder, or behavior the bracketed term may be referring to, include that reference in your synonyms.
The specific term to find synonyms for will be surrounded by double-brackets, like {{term}}.

Please find synonyms given this context: {context}
"""

prompt = PromptTemplate(template=template, input_variables=["context","syn_count"])
llm = ChatOpenAI(model_name=os.getenv('OPENAI_MODEL_NAME'), temperature=0.2)
llm_chain = LLMChain(prompt=prompt, llm=llm)

regex = re.compile('{{(.*?)}}')

def main():
    concepts = get_concepts()
    data, docs = get_data(concepts, errored_only=False)

    # Get total estimated input cost
    #for model in ['gpt-3.5-turbo-1106', 'gpt-4-1106-preview']:
    #    total_estimated_costs, total_tokens = estimate_total_cost(data, docs, model)
    #    print(f'Model: {model}. Total tokens: {total_tokens}. Total estimated costs: ${total_estimated_costs}')

    # Model: gpt-3.5-turbo-1106. Total tokens: 6400923. Total estimated costs: $6.400923
    # Model: gpt-4-1106-preview. Total tokens: 6400923. Total estimated costs: $64.00923

    #num_examples = 1
    #examples = [v for _,v in data.items()]
    #random.shuffle(examples)
    #examples = examples[:num_examples]

    outpath = f'./data/synonyms/medmentions_{os.getenv("OPENAI_MODEL_NAME")}_synonym.json'
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
            response = llm_chain.run(context=context, syn_count=syn_count)

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


def estimate_total_cost(data, docs, model):
    data = [template.replace('{context}', get_context(d, docs[d['filename']])) for _,d in data.items()]
    return get_total_embeddings_cost(data, model)


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
        cur.execute('SELECT cui, str FROM umls')
        for row in cur.fetchall():
            cui, text = row[0], row[1]
            if cui in data:
                data[cui].append(text)
            else:
                data[cui] = [text]
    return data


# Adapted from https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector/
# Helper func: calculate number of tokens
def num_tokens_from_string(string: str, encoding_name = "cl100k_base") -> int:
    if not string:
        return 0
    # Returns the number of tokens in a text string
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# Helper function: calculate cost of embedding num_tokens
# Assumes we're using the text-embedding-ada-002 model
# See https://openai.com/pricing
def get_embedding_cost(num_tokens, model):
    cost_per_1k_tokens = 0 
    if model == 'gpt-4-1106-preview':
        cost_per_1k_tokens = 0.01 
    elif model == 'gpt-3.5-turbo-1106':
        cost_per_1k_tokens = 0.001

    return num_tokens/1000*cost_per_1k_tokens


# Helper function: calculate total cost of embedding all content in the dataframe
def get_total_embeddings_cost(texts, model):
    total_tokens = 0
    for text in texts:
        token_len = num_tokens_from_string(text)
        total_tokens = total_tokens + token_len
    total_cost = get_embedding_cost(total_tokens, model)
    return total_cost, total_tokens


main()
