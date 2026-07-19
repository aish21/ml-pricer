import streamlit as st

from app.ui.theme import apply_workspace_theme
from app.ui.workspace import render_workspace


st.set_page_config(
    page_title="ML Pricer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_workspace_theme()
render_workspace()
