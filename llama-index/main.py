import os
import requests
from pathlib import Path
from llama_index import GPTSimpleVectorIndex, NotionPageReader
from llama_index.readers import notion
from llama_index.readers.schema.base import Document
from llama_index.indices.query.schema import QueryMode


def main(database_id, index_path=None):
    index: GPTSimpleVectorIndex
    if index_path and Path(index_path).exists():
        print(f"Loading index from {index_path}")
        index = GPTSimpleVectorIndex.load_from_disk(index_path)
    else:
        print(f"Loading data from the database {database_id}")
        loader = NotionPageReader(os.environ['NOTION_TOKEN'])
        text = ""
        cursor = None
        page_num = 0
        while True:
            page_num += 1
            print(f"Loading page #{page_num}")
            resp = requests.post(
                notion.DATABASE_URL_TMPL.format(database_id=database_id),
                headers=loader.headers,
                json={"start_cursor": cursor} if cursor else {},
            )
            for page in resp.json()["results"]:
                if (d := page["properties"]["Date"].get("date")) and (rt := page["properties"]["Journal"]["rich_text"]):
                    date = d["start"]
                    journal = rt[0]["text"]["content"]
                    text += f"Date: {date}\n\nJournal:\n{journal}\n\n"
            if not (cursor := resp.json().get("next_cursor")):
                break
        docs=[Document(text)]

        print("Indexing documents...")
        index = GPTSimpleVectorIndex(docs)
        index.save_to_disk(index_path)

    query = 'What I was doing on Mar 13, 2022?'
    print("Querying the index:", query)
    resp = index.query(query)
    print(resp)


if __name__ == "__main__":
    database_id = "8405f70a85b44fe7a211a0a56c0d4cc4"
    index_path = "saves/days_db_index_single_doc"
    main(database_id, index_path=index_path)
