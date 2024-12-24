import os
import logging
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import chainlit as cl


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
VECTOR_DIR = os.environ["VECTOR_DIR"]


system_prompt = (
    "あなたは質問応答タスクのアシスタントです。取得した以下のコンテキストを使用して質問に答えてください。"
    "回答が記載されているURLもわかるように回答を作成してください。答えがわからない場合は、わからないと答えてください。"
    "\n\n"
    "{context}"
)
# プロンプトテンプレート作成
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# LLMのモデル作成
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, streaming=True)

# chain 作成
chain = create_stuff_documents_chain(llm, prompt)

# 類似検索 VectorStoreRetriever 作成
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="私は開発マネージャーの意思決定ログを学習しています。何でも聞いてください。")
    await msg.send()

    # chain をセッションに保存
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    # chain をセッションから取得
    ch = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    mes = "質問は「" + message.content + "」です。"
    try:
        # chain を呼び出す
        res = await ch.ainvoke({"input": mes, "context": retriever.invoke(message.content)}, callbacks=[cb])
    except Exception as e:
        logging.error("通信エラーが発生しました。" + str(e))
        message = "通信エラーが発生しました。"
        await cl.Message(content=message).send()
        return
    
    # 結果をレスポンス
    await cl.Message(content=res).send()
