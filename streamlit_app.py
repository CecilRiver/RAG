import streamlit as st


#页面导航
pg = st.navigation([st.Page("app.py", title="Extraction"), st.Page("history.py", title = "Result"), st.Page("knowledge.py", title = "KnowledgeBase")])
pg.run( )