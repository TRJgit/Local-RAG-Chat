import streamlit as st

st.set_page_config(page_title="Local RAG Chat")

chat_page = st.Page(
    "pages/chat.py", 
    title="Local Rag Chatbot",  
    default=True
)
settings_page = st.Page(
    "pages/instructions.py", 
    title="Prompt Configuration", 
)

pg = st.navigation([chat_page, settings_page])
pg.run()