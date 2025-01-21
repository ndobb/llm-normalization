import os
import json
import psycopg2


data_path = os.path.join('./data', 'synonyms', 'embeddings_cache.json')


with open(data_path, 'r', encoding='utf-8') as fin:
    data = json.loads(fin.read())


def get_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="vectorexample",
        user="postgres",
        password=""
    )

def insert_vectordb(conn, synonym, embedding, cui_id, cui, cui_text):
    sql = 'INSERT INTO synonym_embeddings (synonym, embedding, cui_id, cui, cui_text) SELECT %s, %s, %s, %s, %s'
    cur = conn.cursor()
    cur.execute(sql, (synonym, embedding, cui_id, cui, cui_text))

def check_in_db(conn, synonym):
    sql = 'SELECT 1 from synonym_embeddings WHERE synonym = %s'
    cur = conn.cursor()
    cur.execute(sql, (synonym,))
    for _ in cur.fetchall():
        return True
    return False

with get_db_conn() as conn:
    for synonym, d in data.items():
        if not check_in_db(conn, synonym):
            embedding = d['embedding']
            if any(d['found']):
                cui_id, cui, cui_text = d['found'][0]['id'], d['found'][0]['cui'], d['found'][0]['text']
            else:
                cui_id, cui, cui_text = None, None, None
            insert_vectordb(conn, synonym, embedding, cui_id, cui, cui_text)
            conn.commit()