import os
import json
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector


data_path = os.path.join('./data', 'umls_str_with_embeddings.json')
with open(data_path, 'r', encoding='utf-8') as fin:
    data = json.loads(fin.read())


def divide_chunks(l, n):
    for i in range(0, len(l), n):  
        yield l[i:i + n]


def get_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="umls",
        user="postgres",
        password=""
    )

insert_batch_size = 1000
with get_db_conn() as conn:
    register_vector(conn)
    cur = conn.cursor()
    for i, batch in enumerate(divide_chunks(data['data'], insert_batch_size), 1):
        print(f'Inserting records {i * insert_batch_size}')
        # Prepare the list of tuples to insert
        data_list = [(d['cui'], d['str'], json.dumps(d['sty']), (json.dumps(d['embeddings']) if 'embeddings' in d else None)) for d in batch]
        # Use execute_values to perform batch insertion
        execute_values(cur, "INSERT INTO concept_strs (cui, str, stys, embeddings) VALUES %s", data_list)
        # Commit after we insert all embeddings
        conn.commit()