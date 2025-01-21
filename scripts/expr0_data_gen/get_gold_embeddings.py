import os
import json
import psycopg2
from openai import OpenAI



def main():
    client = OpenAI(api_key=get_openai_key())
    conn = get_db_conn()
    annotations = get_gold_annotations()

    for i, a in enumerate(annotations):
        if check_in_db(conn, a['id']):
            continue
        print(f'Processing {i} of {len(annotations)}...')
        embedding = get_existing_embedding(conn, a['text'])
        if not embedding:
            embedding = get_embedding(client, a['text'])
            embedding = json.dumps(embedding)
        insert_vectordb(conn, a['id'], a['cui'], a['text'], embedding)
        conn.commit()


def get_gold_annotations():
    with open(os.path.join('./data', 'medmentions.json'), 'r', encoding='utf-8') as fin:
        medmentions = json.loads(fin.read())

    annotations = []
    for _, v in medmentions['data']['test'].items():
        for ann in v['annotations']:
            ann['id']       = v['filename'] + '_' + str(ann['indices'][0])
            ann['doc_text'] = v['text']
            ann['filename'] = v['filename']
            annotations.append(ann)

    return annotations


def get_gold_data(conn):
    annotations = []
    cur = conn.cursor()
    cur.execute('SELECT id, cui, str FROM annotation WHERE embedding IS NULL')
    for row in cur.fetchall():
        id, cui, name = row[0], row[1], row[2]
        entry = { 'id': id, 'cui': cui, 'str': name }
        annotations.append(entry)

    return annotations


def check_in_db(conn, id):
    sql = 'SELECT id from annotation WHERE id = %s'
    cur = conn.cursor()
    cur.execute(sql, (id,))
    for _ in cur.fetchall():
        return True
    return False


def insert_vectordb(conn, id, cui, str, embedding):
    sql = 'INSERT INTO annotation (id, cui, str, embedding) SELECT %s, %s, %s, %s'
    cur = conn.cursor()
    cur.execute(sql, (id, cui, str, embedding))


def update_vectordb(conn, id, embeddings):
    sql = 'UPDATE annotation SET embedding = %s WHERE id = %s'
    cur = conn.cursor()
    cur.execute(sql, (embeddings, id))


def get_embedding(client, text, model='text-embedding-3-small'):
    text = text.replace('\n', '')
    return client.embeddings.create(input = [text], model=model).data[0].embedding


def get_openai_key():
    key = ''
    with open('./.env', 'r') as fin:
        for line in fin.readlines():
            if line.startswith('OPENAI_API_KEY'):
                key = line.replace('OPENAI_API_KEY=','').strip()
    return key


def get_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="vectorexample",
        user="postgres",
        password="postgres"
    )


def get_existing_embedding(conn, text):
    sql = 'SELECT embedding FROM umls WHERE str = %s AND embedding IS NOT NULL'
    cur = conn.cursor()
    cur.execute(sql, (text,))
    row = cur.fetchone()
    if row:
        return row[0]
    

main()    