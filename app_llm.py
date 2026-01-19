import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


# 画面に会話履歴を表示
def show_message(message: BaseMessage) -> None:
    if isinstance(message, HumanMessage):
        st.chat_message("human").write(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("ai").write(message.content)


# メイン
def app() -> None:
    # API-KEY読み込み
    load_dotenv(override=True)

    # ①利用LLMを選択
    model = st.sidebar.selectbox(
        label="利用するLLMを選択",
        options=[
            "anthropic:claude-sonnet-4-5",
            "openai:gpt-5-nano",
            "google_genai:gemini-2.5-flash",
        ],
    )

    # ②プロンプト（LLMに与える指示）準備
    prompt = "業務の質問に根拠付きで端的に回答してください。"

    # ③これらをセットしたらAIエージェント完成！(LLMのみなのでAIエージェントとは呼べませんが)
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=prompt,
    )

    # ④タイトル、グラフの図、ここまでの会話履歴を表示
    st.title("業務上の質問に回答します！")
    st.sidebar.image(agent.get_graph().draw_mermaid_png())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        show_message(message)

    # ⑤ユーザーの入力を受け付け
    if user_input := st.chat_input():
        st.session_state.messages.append(HumanMessage(content=user_input))

        # ⑥AIエージェントに回答文を生成させる
        for s in agent.stream(
            {"messages": st.session_state.messages}, stream_mode="values"
        ):
            show_message(s["messages"][-1])
            st.session_state.messages = s["messages"]


app()
