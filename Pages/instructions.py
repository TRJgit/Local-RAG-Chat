import streamlit as st
import system_prompt as sp

st.set_page_config(page_title="Prompt Settings", page_icon="⚙️")

st.header("⚙️ System Prompt Configuration")
st.info("Changes made here will affect how the AI responds in the chat.")

# Initialize session state for the prompt if not already there
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = sp.system_prompt

# Text area to edit the prompt
new_prompt = st.text_area(
    "Edit System Prompt", 
    value=st.session_state.system_prompt, 
    height=400
)

if st.button("Save Prompt"):
    st.session_state.system_prompt = new_prompt
    st.toast("System Prompt updated for this session!")
