import os
import json
import tiktoken
import psycopg2
from openai import OpenAI


data_path = os.path.join('./data', 'umls_str_with_embeddings.json')


def main():
    client = OpenAI(api_key=get_openai_key())

    conn = get_vector_db_conn()
    data = get_concept_data(conn)

    to_process = [x['str'] for x in data['data'] if 'embedding' not in x]
    total_estimated_costs, total_tokens = get_total_embeddings_cost(to_process)
    print(f'Total records: {len(to_process)}. Total tokens: {total_tokens}. Total estimated costs: ${total_estimated_costs}')
    
    i = 0
    for concepts in batch(data['data'], 300):
        i += 1
        print(f'Batch {i}...')
        strings = [c['str'] for c in concepts]
        ids     = [c['id'] for c in concepts]
        embeddings = get_embedding(client, strings)
        updates = zip(ids, [e.embedding for e in embeddings.data])
        for id, embedding in updates:
            update_vectordb(conn,id, embedding)
        if i % 10:
            conn.commit()
    conn.commit()


def update_vectordb(conn, id, embeddings):
    sql = 'UPDATE umls SET embedding = %s WHERE id = %s'
    cur = conn.cursor()
    cur.execute(sql, (json.dumps(embeddings), id))


def get_concept_data(conn):
    data = {'data': []}
    cur = conn.cursor()
    cur.execute('SELECT id, cui, str, sty FROM umls WHERE embedding IS NULL')
    for row in cur.fetchall():
        id, cui, name, stys = row[0], row[1], row[2], row[3]
        entry = { 'id': id, 'cui': cui, 'str': name, 'stys': stys }
        data['data'].append(entry)

    return data


def get_vector_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="vectorexample",
        user="postgres",
        password=""
    )


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def get_embedding(client, texts, model='text-embedding-3-small'):
    return client.embeddings.create(input = texts, model=model)


def get_openai_key():
    key = ''
    with open('./.env', 'r') as fin:
        for line in fin.readlines():
            if line.startswith('OPENAI_API_KEY'):
                key = line.replace('OPENAI_API_KEY=','').strip()
    return key


# Below adapted from https://www.timescale.com/blog/postgresql-as-a-vector-database-create-store-and-query-openai-embeddings-with-pgvector/
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
def get_embedding_cost(num_tokens):
    return num_tokens/1000*0.00002


# Helper function: calculate total cost of embedding all content in the dataframe
def get_total_embeddings_cost(texts):
    total_tokens = 0
    for text in texts:
        token_len = num_tokens_from_string(text)
        total_tokens = total_tokens + token_len
    total_cost = get_embedding_cost(total_tokens)
    return total_cost, total_tokens
            

main()            