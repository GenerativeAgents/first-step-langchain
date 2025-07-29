.PHONY: streamlit
streamlit:
	uv run python -m streamlit run app.py --server.port 8080

.PHONY: streamlit_llm
streamlit_llm:
	uv run python -m streamlit run app_llm.py --server.port 8080