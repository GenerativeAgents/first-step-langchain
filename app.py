import streamlit as st
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent


# 画面に会話履歴を表示
def show_message(message: BaseMessage) -> None:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        if message.tool_calls:
            for tool_call in message.tool_calls:
                query = tool_call.get("args", {}).get("query", "")
                if "tavily" in tool_call["name"]:
                    st.chat_message("ai").write(f"（ネットで「{query}」を検索中...）")
                elif "local" in tool_call["name"]:
                    st.chat_message("ai").write(f"（社内資料で「{query}」を検索中...）")
        else:
            st.chat_message("ai").write(message.content)


# ファイルを読み込んでベクトル検索DBを構築
@st.cache_resource  # アプリ起動後の初回のみ
def create_local_search_tool() -> Tool:
    documents = DirectoryLoader("./docs", glob="**/*").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(
        texts, OpenAIEmbeddings(model="text-embedding-3-small")
    )
    return create_retriever_tool(
        vectorstore.as_retriever(), name="local_search", description="社内資料の検索"
    )


# メイン
def app() -> None:
    # API-KEY読み込み
    load_dotenv(override=True)

    # ①ネット検索ツールの準備
    web_search_tool = TavilySearch(max_results=5)

    # ②ファイルの検索ツールの準備
    local_search_tool = create_local_search_tool()

    # ③利用LLMを選択
    model = st.sidebar.selectbox(
        label="利用するLLMを選択",
        options=[
            "anthropic:claude-sonnet-4-0",
            "openai:gpt-4.1-mini",
            "google_genai:gemini-2.5-flash",
        ],
    )

    # ④プロンプト（LLMに与える指示）準備
    prompt = "社内資料に基づき、業務の質問に根拠付きで回答してください。必要に応じてネットも検索してください。"

    # ⑤これらをセットしたらAIエージェント完成！
    agent = create_react_agent(
        model=model,
        tools=[web_search_tool, local_search_tool],
        prompt=prompt,
    )

    # ⑥タイトル、グラフの図、ここまでの会話履歴を表示
    st.title("業務上の質問に回答します！")
    st.sidebar.image(agent.get_graph().draw_mermaid_png())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        show_message(message)

    # ⑦ユーザーの入力を受け付け
    if user_input := st.chat_input():
        st.session_state.messages.append(HumanMessage(content=user_input))

        # ⑧AIエージェントに回答文を生成させる
        for s in agent.stream(
            {"messages": st.session_state.messages}, stream_mode="values"
        ):
            show_message(s["messages"][-1])
            st.session_state.messages = s["messages"]


app()
