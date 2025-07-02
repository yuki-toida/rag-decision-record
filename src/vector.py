import logging
import requests
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# 定数
NOTION_API_VERSION = "2022-06-28"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 0

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 環境変数ロード
default_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=default_env_path if default_env_path.exists() else None)

class NotionVectorizerError(Exception):
    pass

class NotionVectorizer:
    def __init__(self):
        try:
            self.notion_api_token = os.environ["NOTION_API_TOKEN"]
            self.notion_database_id = os.environ["NOTION_DATABASE_ID"]
            self.notion_dir = Path(os.environ["NOTION_DIR"])
            self.vector_dir = Path(os.environ["VECTOR_DIR"])
        except KeyError as e:
            raise NotionVectorizerError(f"環境変数が不足しています: {e}")
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {self.notion_api_token}",
            "Notion-Version": NOTION_API_VERSION
        }

    def fetch_notion_database(self) -> Dict[str, Any]:
        payload = {'page_size': 100}
        url = f"https://api.notion.com/v1/databases/{self.notion_database_id}/query"
        response = requests.post(url, headers=self.headers, json=payload).json()
        while response.get('has_more'):
            payload['start_cursor'] = response.get('next_cursor')
            next_page = requests.post(url, headers=self.headers, json=payload).json()
            response['results'].extend(next_page['results'])
            response['has_more'] = next_page['has_more']
            response['next_cursor'] = next_page.get('next_cursor')
        return response

    def fetch_notion_page_children(self, page_id: str) -> Dict[str, Any]:
        page_url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        return requests.get(page_url, headers=self.headers).json()

    def extract_text_from_page(self, result: Dict[str, Any]) -> str:
        page_id = result["id"].replace("-", "")
        page_response = self.fetch_notion_page_children(page_id)
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

    def save_texts_to_files(self, texts: List[str]) -> None:
        self.notion_dir.mkdir(exist_ok=True)
        for idx, text in enumerate(texts, 1):
            file_path = self.notion_dir / f"{idx}.txt"
            with file_path.open(mode="w", encoding="utf-8") as f:
                f.write(text)
            logging.info(f"Saved: {file_path}")

    def split_documents_from_files(self) -> List[Any]:
        documents = []
        files = list(self.notion_dir.glob("*.txt"))
        for file in files:
            loader = TextLoader(str(file), encoding="utf-8")
            page = loader.load()
            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            documents += text_splitter.split_documents(page)
        return documents

    def save_documents_to_faiss(self, documents: List[Any]) -> None:
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(str(self.vector_dir))
        logging.info(f"FAISS DB saved to: {self.vector_dir}")

    def run(self) -> None:
        try:
            response = self.fetch_notion_database()
            texts = [self.extract_text_from_page(result) for result in response["results"]]
            self.save_texts_to_files(texts)
            documents = self.split_documents_from_files()
            self.save_documents_to_faiss(documents)
            logging.info("ベクトルDBの作成が完了しました。")
        except Exception as e:
            logging.error(f"エラーが発生しました: {e}")
            raise NotionVectorizerError(e)

if __name__ == "__main__":
    NotionVectorizer().run()
