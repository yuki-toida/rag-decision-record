import os
import glob
import requests
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 定数
NOTION_API_VERSION = "2022-06-28"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 0

NOTION_API_TOKEN = os.environ["NOTION_API_TOKEN"]
NOTION_DATABASE_ID = os.environ["NOTION_DATABASE_ID"]
NOTION_DIR = os.environ["NOTION_DIR"]
VECTOR_DIR = os.environ["VECTOR_DIR"]


def fetch_notion_database() -> Dict[str, Any]:
    """Notionデータベースの全ページ情報を取得する"""
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {NOTION_API_TOKEN}",
        "Notion-Version": NOTION_API_VERSION
    }
    payload = {'page_size': 100}
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    response = requests.post(url, headers=headers, json=payload).json()
    while response.get('has_more'):
        payload['start_cursor'] = response.get('next_cursor')
        next_page = requests.post(url, headers=headers, json=payload).json()
        response['results'].extend(next_page['results'])
        response['has_more'] = next_page['has_more']
        response['next_cursor'] = next_page.get('next_cursor')
    return response

def fetch_notion_page_children(page_id: str, headers: Dict[str, str]) -> Dict[str, Any]:
    """Notionページの子ブロックを取得する"""
    page_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
    return requests.get(page_url, headers=headers).json()

def extract_text_from_page(result: Dict[str, Any], headers: Dict[str, str]) -> str:
    """Notionページ情報から本文テキストを抽出する"""
    page_id = result["id"].replace("-", "")
    page_response = fetch_notion_page_children(page_id, headers)
    title = result["properties"]["Title"]["title"][0]["plain_text"]
    root_url = f"https://www.notion.so/{page_id}"
    body = f"タイトルは、{title}。参照リンクは、{root_url}。"
    for page in page_response["results"]:
        for key in ["paragraph", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
            if key in page:
                texts = page[key]["rich_text"]
                if texts:
                    body += str.strip(texts[0]["plain_text"]) + " "
                break
    return body.replace("\n", "")

def save_texts_to_files(texts: List[str], output_dir: str) -> None:
    """テキストリストを個別ファイルとして保存する"""
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for idx, text in enumerate(texts, 1):
        with open(f"{output_dir}/{idx}.txt", mode="w", encoding="utf-8") as f:
            f.write(text)

def split_documents_from_files(input_dir: str) -> List[Any]:
    """テキストファイル群を分割し、LangChain用ドキュメントリストに変換する"""
    documents = []
    files = glob.glob(input_dir + "/*.txt")
    for file in files:
        loader = TextLoader(file, encoding="utf-8")
        page = loader.load()
        text_splitter = CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        documents += text_splitter.split_documents(page)
    return documents

def save_documents_to_faiss(documents: List[Any], vector_dir: str) -> None:
    """ドキュメントリストをFAISSベクトルDBとして保存する"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local(vector_dir)

def main() -> None:
    """Notionデータベースからデータを取得し、ベクトルDBを作成するメイン処理"""
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {NOTION_API_TOKEN}",
        "Notion-Version": NOTION_API_VERSION
    }
    try:
        response = fetch_notion_database()
        texts = [extract_text_from_page(result, headers) for result in response["results"]]
        save_texts_to_files(texts, NOTION_DIR)
        documents = split_documents_from_files(NOTION_DIR)
        save_documents_to_faiss(documents, VECTOR_DIR)
        print("ベクトルDBの作成が完了しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
