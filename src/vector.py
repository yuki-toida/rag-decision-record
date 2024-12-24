import requests
import os
import glob
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

NOTION_API_TOKEN = os.environ["NOTION_API_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
NOTION_DIR = os.environ["NOTION_DIR"]
VECTOR_DIR = os.environ["VECTOR_DIR"]

def main():
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {NOTION_API_TOKEN}",
        "Notion-Version": "2022-06-28"
    }
    payload = {'page_size': 100}

    # Notion DB 取得
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    response = requests.post(url, headers=headers, json=payload).json()

    while response.get('has_more'):
        payload['start_cursor'] = response.get('next_cursor')
        next_page = requests.post(url, headers=headers, json=payload).json()
        response['results'].extend(next_page['results'])
        response['has_more'] = next_page['has_more']
        response['next_cursor'] = next_page.get('next_cursor')

    count = 0
    for result in response["results"]:
        count += 1

        page_id = result["id"].replace("-", "")

        # Notion 子ページ取得
        page_url = "https://api.notion.com/v1/blocks/" + page_id + "/children"
        page_response = requests.get(page_url, headers=headers).json()

        # タイトルと参照リンクを本文に含める
        title = result["properties"]["Title"]["title"][0]["plain_text"]
        root_url = "https://www.notion.so/" + page_id
        body = f"タイトルは、{title}。参照リンクは、{root_url}。"

        for page in page_response["results"]:
            if "heading_1" in page:
                continue

            if "paragraph" in page:
                texts = page["paragraph"]["rich_text"]
                if len(texts) == 0:
                    continue
                body += str.strip(texts[0]["plain_text"]) + " "
            elif "heading_2" in page:
                texts = page["heading_2"]["rich_text"]
                if len(texts) == 0:
                    continue
                body += str.strip(texts[0]["plain_text"]) + " "
            elif "heading_3" in page:
                texts = page["heading_3"]["rich_text"]
                if len(texts) == 0:
                    continue
                body += str.strip(texts[0]["plain_text"]) + " "
            elif "bulleted_list_item" in page:
                texts = page["bulleted_list_item"]["rich_text"]
                if len(texts) == 0:
                    continue
                body += str.strip(texts[0]["plain_text"]) + " "
            elif "numbered_list_item" in page:
                texts = page["numbered_list_item"]["rich_text"]
                if len(texts) == 0:
                    continue
                body += str.strip(texts[0]["plain_text"]) + " "

        body = body.replace("\n", "")

        if not os.path.isdir(NOTION_DIR):
            os.mkdir(NOTION_DIR)
        
        # テキストファイルとして保存
        with open(f"{NOTION_DIR}/{count}.txt", mode="w") as f:
            f.write(body)

    # テキストファイルを分割
    documents = []
    files = glob.glob(NOTION_DIR + "/*.txt")
    for file in files:
        loader = TextLoader(file, encoding="utf-8")
        page = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )
        documents += text_splitter.split_documents(page)

    # ベクトルDBとして保存
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(VECTOR_DIR)


if __name__ == "__main__":
    main()
