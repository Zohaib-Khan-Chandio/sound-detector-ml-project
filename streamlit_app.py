"""
Environmental Sound Detector - Streamlit version (same web page design, same trained model).

The page you see is frontend/index.html, shown as a Streamlit custom component.
The browser decodes the audio and sends it here; this script computes the same 316 features
as in training, runs the trained model, and sends the answer back to the page.
"""
import os

import streamlit as st
import streamlit.components.v1 as components

from core import load_bundle, predict_b64

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Environmental Sound Detector", page_icon="🔊",
                   layout="wide", initial_sidebar_state="collapsed")

# make Streamlit's own frame disappear so only the website design is visible
st.markdown("""
<style>
#MainMenu, header, footer, [data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none !important;}
.block-container, [data-testid="stMainBlockContainer"], [data-testid="stAppViewBlockContainer"]
{padding:0 !important; max-width:100% !important;}
[data-testid="stVerticalBlock"] {gap:0 !important;}
[data-testid="stAppViewContainer"], .stApp {background:transparent !important;}
iframe {border:0 !important; width:100% !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the trained model...")
def get_bundle():
    return load_bundle(HERE)


try:
    bundle = get_bundle()
except Exception as e:  # clear message instead of a crash
    st.error(f"The model could not be loaded: {e}")
    st.stop()

ui = components.declare_component("sound_ui", path=os.path.join(HERE, "frontend"))

info = {"model": bundle["name"], "threshold": bundle["threshold"]}
value = ui(info=info, prediction=st.session_state.get("pred"), key="ui", default=None)

# a new audio clip arrived from the page
if value and value.get("id") != st.session_state.get("last_id"):
    st.session_state["last_id"] = value["id"]
    try:
        pred = predict_b64(bundle, value["pcm_b64"])
    except Exception as e:
        pred = {"error": str(e)}
    pred["id"] = value["id"]
    st.session_state["pred"] = pred
    st.rerun()
