import streamlit as st

st.set_page_config(page_title="Prompt Settings")
st.header("System Prompt Configuration")
st.info("Changes made here will affect how the AI responds in the chat.")

def read_system_prompt():
    with open("system_prompt.txt", "r") as f:
        return f.read()

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = read_system_prompt()

new_prompt = st.text_area(
    "Edit System Prompt", 
    value=st.session_state.system_prompt, 
    height=400
)

if st.button("Save Prompt"):
    st.session_state.system_prompt = new_prompt
    with open("system_prompt.txt", "w") as f:
        f.write(new_prompt)
    st.toast("System Prompt updated for this session!")
