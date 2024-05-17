import streamlit as st
# import final_test
import llama3
# import echo


def answer(question):
    # anything
    return llama3.answer(question)


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

#  command to run:
#  streamlit run \path\main.py

# or
# streamlit run main.py