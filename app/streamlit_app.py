"""Entry point: `streamlit run app/streamlit_app.py` from the repository root."""
import streamlit as st

st.set_page_config(page_title="Ethiopia Macro Shock Platform", page_icon="📈", layout="wide")

pages = st.navigation([
    st.Page("pages/overview.py", title="Overview", icon="📊", default=True),
    st.Page("pages/shock_transmission.py", title="Shock transmission", icon="🛢️"),
])
pages.run()