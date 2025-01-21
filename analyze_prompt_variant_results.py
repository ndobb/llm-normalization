import os
import json
import re

datasets = [
    # Expr 2
    #['vicuna_3b', os.path.join('./data', 'medmentions_vicuna13b_prompt_variants_by_source_1000.json')],
    #['gpt3.5_3b', os.path.join('./data', 'medmentions_gpt3.5_prompt_variants_by_source.json')],

    #['vicuna_3a_en_cn', os.path.join('./data', 'medmentions_vicuna13b_prompt_variants_yesno_en_cn.json')],

    # Expr 3
    #['vicuna_1000', os.path.join('./data', 'medmentions_vicuna13b_prompt_variants_yesno_1000.json')],
    #['vicuna_1000', os.path.join('./data', 'medmentions_vicuna13b_prompt_variants_by_source_1000.json')]

    # Expr 4
    ['vicuna', os.path.join('./data', 'medmentions_vicuna13b_prompt_variants_yesno.json')],
    ['gpt3.5', os.path.join('./data', 'medmentions_gpt3.5_prompt_variants_yesno.json')],

    
]

def main():
    stats = []
    sources = ['metamap', 'bm25', 'quickumls','embedding']
    for dataset_name, dataset_path in datasets:
        with open(dataset_path, 'r', encoding='utf-8') as fout:
            results = json.loads(fout.read())

        experiment_types = list(results[0]['responses'].keys())

        # Hack to add in source-specific calculations
        for exp_type in experiment_types:
            if 'yesno' in exp_type:
                if exp_type in results[0]['responses'] and 'sources' in results[0]['responses'][exp_type]['responses'][0]:
                    experiment_types += [f'{exp_type}|{s}' for s in sources]

        for _exp_type in experiment_types:

            for take_top in [0,1]:
                for strict_k in [False]:
                    if strict_k and 'yesno' in exp_type:
                        continue
                    if '|' in _exp_type:
                        source = _exp_type.split('|')[1]
                        if 'yesno' in _exp_type:
                            exp_type = _exp_type.split('|')[0]
                        else:
                            exp_type = _exp_type
                    else:
                        exp_type = _exp_type
                        source = 'all'
                    
                    for row in results:

                        if 'responses' in row and exp_type in row['responses']:
                            val = row['responses'][exp_type]

                            if 'yesno' in exp_type:
                                val['correct']  = val['answer']
                                val['response'] = val['responses']

                            if source != 'all' and 'yesno' in exp_type:
                                val['response'] = [r for r in val['response'] if 'sources' not in r or source in r['sources']]

                            answers, reason = parse_response(dataset_name, exp_type, val['response'], take_top, strict_k)

                            if 'idx' in exp_type or 'cui' in exp_type:
                                direct   = val['choices']['direct']
                                indirect = val['choices']['indirect']

                                # Take only single option if that's all available
                                if len(direct) + len(indirect) == 1:
                                    answers = [parse_string_answer((direct + indirect)[0])]

                                # Auto-include first answer
                                elif take_top and (any(direct) or any(indirect)):
                                    for choice in (direct + indirect)[:take_top]:
                                        answer = parse_string_answer(choice)
                                        answers.append(answer)

                            val['answers'] = list(set(answers))

                            val['matched'] = val['correct'] in answers
                            val['reason']  = reason

                    # Recall
                    r_num   = len([x for x in results if 'responses' in x and exp_type in x['responses'] and x['responses'][exp_type]['matched']])
                    r_denom = len([x for x in results if 'responses' in x and exp_type in x['responses']])
                    r = round(r_num * 1.0 / r_denom * 100, 1)

                    # Precision
                    p_num   = len([x for x in results if 'responses' in x and exp_type in x['responses'] and x['responses'][exp_type]['matched']])
                    p_denom = sum([len(x['responses'][exp_type]['answers']) for x in results if 'responses' in x and exp_type in x['responses']])
                    p = round(p_num * 1.0 / p_denom * 100, 1)

                    # F1
                    β = 2
                    f1 = round(2 * (p * r) / (p + r), 1)
                    fβ = round((1 + β**2) * (p * r) / ((β**2 * p) + r), 1)

                    n = f'(n={r_denom})'
                    top = f'(auto top {take_top})' if take_top else ''
                    strict = '(strict)' if strict_k else ''
                    src = f'({source})' if source != 'all' else ''

                    for s in sources + ['|', '_flex']:
                        exp_type = exp_type.replace(s, '')

                    to_print = f'{dataset_name:<17} {exp_type:<15} {src:<12} {top:<13} {strict:<10} {n:<10} R: {r:<7} P: {p:<7} F1: {f1:<7} Fβ: {fβ:<7}'

                    stats.append([r, p, f1, fβ, to_print])
    
    for _, _, _, _, stat in sorted(stats, key=lambda x: x[3], reverse=True):

        print(stat)


re_response_json = re.compile('({\"answer\":[^}]*})')
re_cuis = re.compile('(C\d+)')
re_indices = re.compile('(\d+)')
re_k = re.compile('_([1-3])_?')


def parse_string_answer(answer_0):
    return answer_0.split(' ')[0].replace('(','').replace(')','').strip()


def parse_response(dataset_name, experiment_type, response, take_top, strict_k):
    if not response:
        return [], ''

    # Auto-include first answer
    if 'yesno' in experiment_type:
        if 'cn' in dataset_name:
            return [x['answer'] for i,x in enumerate(response) if i < take_top or ('yes' in x['response'].lower() or '是的' in x['response'])], ''
        return [x['answer'] for i,x in enumerate(response) if i < take_top or 'true' in x['response'].lower()], ''
    
    if strict_k:
        k = re_k.search(experiment_type)
        k = int(k.groups(1)[0]) if k else 10
    else:
        k = 10

    if 'idx' in experiment_type:
        return parse_response_indices(response, k)
    return parse_response_cuis(response, k)


def parse_response_indices(response, k):
    structured = None
    orig_response = response
    answers, reason = [], ''
    idxs = re_indices.findall(response)
    response = response.replace('\n','')

    if '"reason": "' in response and '"}' not in response:
        response += '"}'

    try:
        structured = json.loads(response)
    except:
        pass

    cg = re_response_json.findall(response)
    if cg:
        try:
            for hit in cg:
                structured = json.loads(hit)
                if 'answer' in structured:
                    answers += structured['answer']
                if 'reason' in structured:
                    reason += structured['reason']
        except:
            pass

    if structured:
        if type(structured) == int:
            answers = [structured]
        elif type(structured) == dict:
            if 'answer' in structured:
                answers = structured['answer']
            if 'reason' in structured:
                reason = structured['reason']

    else:
        temp_response = orig_response.lower()
        beg_idx = temp_response.find('the answer is')
        if beg_idx > -1:
            end_idx = temp_response[beg_idx:].find('.')
            temp_response = temp_response[beg_idx:end_idx].replace('{','[').replace('}',']')
            try:
                answers = json.loads(temp_response)
            except:
                pass

    if type(answers) == int:
        answers = [answers]
    if idxs and any(idxs) and not any(answers):
        answers = list(idxs)

    answers = list(dict.fromkeys([str(a) for a in answers]))
    return answers[:k], reason


def parse_response_cuis(response, k):
    structured = None
    orig_response = response
    answers, reason = [], ''
    cuis = re_cuis.findall(response)
    if cuis:
        for cui in cuis:
            response = response.replace(cui, '"' + cui + '"')
    response = response.replace('""', '"').replace('""', '"')
    response = response.replace('\n','')

    if '"reason": "' in response:
        offset = len('"reason": "')
        reason_beg_idx = response.find('"reason": "')
        response = response[:reason_beg_idx] + '"reason": "' + response[reason_beg_idx+offset:-2].replace('"', "\'")
        if '"}' not in response:
            response += '"}'

    try:
        structured = json.loads(response.replace('""', '"'))
    except:
        pass

    cg = re_response_json.search(response.replace('""', '"'))
    if cg:
        try:
            structured = json.loads(cg.group(1))
        except:
            pass

    if structured and type(structured) == dict:
        if 'answer' in structured:
            answers = structured['answer']
        if 'reason' in structured:
            reason = structured['reason']

    else:
        temp_response = orig_response.lower()
        beg_idx = temp_response.find('the answer is')
        if beg_idx > -1:
            end_idx = temp_response[beg_idx:].find('.')
            temp_response = temp_response[beg_idx:end_idx].replace('{','[').replace('}',']')
            try:
                answers = json.loads(temp_response)
            except:
                pass

    if cuis and any(cuis) and not any(answers):
        answers = list(cuis)

    answers = list(dict.fromkeys([a for a in answers if len(str(a)) == 8]))

    return answers[:k], reason

    
main()