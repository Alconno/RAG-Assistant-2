import os, warnings, logging

# ---------------------- Irrelevant Warnings ----------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUPY_DISABLE_DEPRECATION_WARNINGS"] = "1"
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("weaviate").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*MessageFactory.*GetPrototype.*")


from core.state import RuntimeState
from services.api import build_api
from app import run_app
import streamlit as st


if __name__ == "__main__":
    if "rs" not in st.session_state:
        rs = RuntimeState()
        rs.api = build_api()
        st.session_state.rs = rs

    rs = st.session_state.rs

    run_app(rs)