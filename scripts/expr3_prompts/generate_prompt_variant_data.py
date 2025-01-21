import os
import sys
import json
from pgvector.psycopg2 import register_vector
from tqdm import tqdm

sys.path.append('/Users/ndobb/work/llm-normalization/')

from quickumls import QuickUMLS
from retriv import SparseRetriever
from scripts.expr1_baseline.get_retrieval_perf_bounds import (
    get_vector_db_conn, get_gold_annotations, get_umls_concepts, get_derived_synonyms,
    get_bm25_unique_cuis, normalize_by_metamap, normalize_by_quickUmls, parse_llm_synonyms,
    get_topk_similar_concepts, get_cached_synonym_embedding
)


def main():
    conn     = get_vector_db_conn()
    test     = get_gold_annotations()
    concepts = get_umls_concepts(conn)
    synonyms = get_derived_synonyms()
    terms    = get_preferred_terms()
    register_vector(conn)

    bm25 = SparseRetriever().index(concepts)
    quickUmls = QuickUMLS('./data/QuickUMLS')
    extractors = [
        [ 'bm25',      lambda text, embedding, k: get_bm25_unique_cuis(bm25, text, k) ],
        [ 'metamap',   lambda text, embedding, k: normalize_by_metamap(text) ],
        [ 'quickumls', lambda text, embedding, k: normalize_by_quickUmls(quickUmls, text) ],
        #[ 'embedding', lambda text, embedding, k: get_topk_similar_concepts(conn, k, embedding) ]
    ]

    concepts = {c['cui']:c for c in concepts}
    k = 5

    for syn_k in [1]:
        for synonym_dataset_name, synonym_dataset in synonyms.items():
            output = []
            for d in tqdm(test):
                out = d
                out['options'] = {}
                #if 'embedding' in out:
                #    del out['embedding']
                embedding = d['embedding'] if 'embedding' in d else None
                for extractor_name, extract in extractors:
                    orig_extractor_name = extractor_name
                    extractor_name += '_' + synonym_dataset_name
                    found = extract(d['text'], embedding, k)

                    for f in found:
                        f['isDirect'] = True
                        f['sources'] = [orig_extractor_name]
                        f['synonym'] = None
                        if f['cui'] not in out['options']:
                            out['options'][f['cui']] = f
                        else:
                            out['options'][f['cui']]['sources'] += [orig_extractor_name]

                    synonym_entry = synonym_dataset.get(out['id'])
                    for synonym in parse_llm_synonyms(synonym_entry['response']):
                        embedding = get_cached_synonym_embedding(synonym) if 'embedding' in extractor_name else None
                        found_syns = extract(synonym, embedding, syn_k)
                        for f in found_syns:
                            f['isDirect'] = False
                            f['sources'] = [orig_extractor_name]
                            f['synonym'] = synonym
                            if f['cui'] not in out['options']:
                                out['options'][f['cui']] = f
                            else:
                                out['options'][f['cui']]['sources'] += [orig_extractor_name]
                    
                    if out['cui'] not in out['options']:
                        c = concepts[out['cui']]
                        out['options'][out['cui']] = { 
                            'id': None, 'cui': out['cui'], 'text': c['text'], 'stys': c['stys'], 'sources': [], 'isDirect': True, 'synonym': None
                        }
        
                out['options'] = [get_simplified_concept(c, concepts, terms) for _,c in out['options'].items() if c['cui'] in concepts]
                output.append(out)

            #with open(os.path.join('./data', f'prompt_variant_test_{synonym_dataset_name}_concepts.json'), 'w+', encoding='utf-8') as fout:
            #    fout.write(json.dumps(output, indent=4))


def get_simplified_concept(c, concepts, preferred_terms):
    text_field = ''
    for field in ['text', 'term', 'name']:
        if field in c:
            text_field = field
            break
    
    if c['cui'] in preferred_terms:
        pt = preferred_terms[c['cui']]['str']
    else:
        pt = None

    return { 
        'cui': c['cui'], 'str': c[text_field], 'stys': concepts[c['cui']]['stys'], 
        'pt': pt, 'isDirect': c['isDirect'], 'sources': list(set(c['sources'])), 'synonym': c['synonym']
    }


def get_preferred_terms():
    terms = {}
    with open(os.path.join('./data', 'umls_preferred_terms.json'), 'r', encoding='utf-8') as fin:
        for term in json.loads(fin.read())['data']:
            terms[term['cui']] = term

    return terms


main()    