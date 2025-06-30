import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chainlit as cl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

default_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=default_env_path if default_env_path.exists() else None)

class RAGApplicationError(Exception):
    pass

class RAGConfig:
    def __init__(self) -> None:
        try:
            self.openai_api_key: str = os.environ["OPENAI_API_KEY"]
            self.vector_dir: str = os.environ["VECTOR_DIR"]
        except KeyError as e:
            raise RAGApplicationError(f"環境変数が不足しています: {e}")
        
        self.model_name: str = "gpt-4o"
        self.temperature: int = 0
        self.embedding_model: str = "text-embedding-3-large"
        self.retriever_k: int = 10
        self.system_prompt: str = (
            "あなたは質問応答タスクのアシスタントです。取得した以下のコンテキストを使用して質問に答えてください。"
            "回答が記載されているURLもわかるように回答を作成してください。答えがわからない場合は、わからないと答えてください。"
            "\n\n"
            "{context}"
        )

class RAGApplication:
    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        self.config: RAGConfig = config or RAGConfig()
        self._chain: Optional[StuffDocumentsChain] = None
        self._retriever: Optional[VectorStoreRetriever] = None
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.config.system_prompt),
                ("human", "{input}"),
            ])
            
            llm = ChatOpenAI(
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                streaming=True
            )
            
            self._chain = create_stuff_documents_chain(llm, prompt)
            
            embeddings = OpenAIEmbeddings(model=self.config.embedding_model)
            vector_store = FAISS.load_local(
                self.config.vector_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            self._retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": self.config.retriever_k}
            )
            
            logging.info("RAGアプリケーションの初期化が完了しました")
        except Exception as e:
            logging.error(f"RAGアプリケーションの初期化中にエラーが発生しました: {e}")
            raise RAGApplicationError(f"初期化エラー: {e}")
    
    @property
    def chain(self) -> Optional[StuffDocumentsChain]:
        return self._chain
    
    @property
    def retriever(self) -> Optional[VectorStoreRetriever]:
        return self._retriever

rag_app = RAGApplication()

@cl.on_chat_start
async def on_chat_start() -> None:
    try:
        msg = cl.Message(content="私は開発マネージャーの意思決定ログを学習しています。何でも聞いてください。")
        await msg.send()
        cl.user_session.set("rag_app", rag_app)
        logging.info("チャットセッションが開始されました")
    except Exception as e:
        logging.error(f"チャットセッション開始中にエラーが発生しました: {e}")
        await cl.Message(content="申し訳ありませんが、システムの初期化中にエラーが発生しました。").send()

@cl.on_message
async def on_message(message: cl.Message) -> None:
    app: Optional[RAGApplication] = cl.user_session.get("rag_app")
    if not app:
        await cl.Message(content="セッションが無効です。チャットを再開してください。").send()
        return
    
    cb = cl.AsyncLangchainCallbackHandler()
    user_question: str = f"質問は「{message.content}」です。"
    
    try:
        context = app.retriever.invoke(message.content)
        response = await app.chain.ainvoke(
            {"input": user_question, "context": context}, 
            callbacks=[cb]
        )
        await cl.Message(content=response).send()
        logging.info(f"質問に回答しました: {message.content[:50]}...")
    except Exception as e:
        error_msg = f"質問の処理中にエラーが発生しました: {str(e)}"
        logging.error(error_msg)
        await cl.Message(content="申し訳ありませんが、質問の処理中にエラーが発生しました。もう一度お試しください。").send()
