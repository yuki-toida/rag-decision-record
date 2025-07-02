# rag-decision-record

## 概要

このプロジェクトは、Notionのデータベースから意思決定記録を取得し、LangChain・OpenAI・FAISSを用いてベクトル化・検索し、ChainlitベースのチャットUIで自然言語質問応答を行うアプリケーションです。

---

## セットアップ手順

1. **uvのインストール**
   uvがインストールされていない場合は、以下のコマンドでインストールしてください：

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **依存パッケージのインストール**

   ```sh
   uv sync
   ```

3. **環境変数の設定**
   `.env` などで以下を設定してください：
   - `OPENAI_API_KEY` : OpenAI APIキー
   - `NOTION_API_TOKEN` : Notion APIトークン
   - `NOTION_DATABASE_ID` : NotionデータベースID
   - `NOTION_DIR` : Notionテキスト出力ディレクトリ（例: output_notion）
   - `VECTOR_DIR` : ベクトルDB出力ディレクトリ（例: output_vector）

---

## ベクトルDBの作成

Notionからデータを取得し、ベクトルDBを作成します。

```sh
uv run python src/vector.py
```

---

## チャットアプリの起動

Chainlitを使ってチャットUIを起動します。

```sh
uv run chainlit run src/app.py
```

起動後、表示されるURLにアクセスしてください。

---

## ライセンス

MIT License
