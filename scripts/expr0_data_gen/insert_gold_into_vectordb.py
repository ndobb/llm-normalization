import os
import json
import psycopg2


data_paths = [
    ['semeval', os.path.join('./data', 'llm_normalization_semeval2014_test.json')],
    ['medmentions', os.path.join('./data', 'llm_normalization_medmentions_test.json')],
]


def main():
    conn = get_db_conn()

    for dataset_name, data_path in data_paths:
        with open(data_path, 'r', encoding='utf-8') as fin:
            data = json.loads(fin.read())
        
        for _, doc in data['data'].items():
            for annotation in doc['annotations']:

                id = f'{doc["filename"]}_{annotation["indices"][0]}'
                cui = annotation['cui']
                text = annotation['text']
                if ' ' in cui:
                    continue
                add_to_vectordb(conn, id, cui, dataset_name, text)
                conn.commit()


def add_to_vectordb(conn, id, cui, dataset, text):
    sql = 'INSERT INTO annotation_embeddings (id, cui, dataset, str) SELECT %s, %s, %s, %s'
    cur = conn.cursor()
    cur.execute(sql, (id, cui, dataset, text))    


def get_db_conn():
    return psycopg2.connect(
        host="localhost",
        database="umls",
        user="postgres",
        password=""
    )


main()