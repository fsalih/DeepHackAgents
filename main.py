import streamlit as st
import final_test


def answer(question):
    # anything
    return final_test.agent(question)


st.title("Deep Hack Winners (DHW) Bot")

if "model" not in st.session_state:
    st.session_state["model"] = "GIGA-CHAT"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        st.markdown('Подожди, готовлю ответ...')
        response = answer(prompt)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

#  command for run:
#  streamlit run \path\main.py