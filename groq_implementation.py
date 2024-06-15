from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

groq_api = os.getenv("GROQ_API")

def main():
    st.title("GROQ Implementation (by Bose)")
    st.sidebar.title("Select the model: ")
    model = st.sidebar.radio("Choose the model", ("gemma-7b-it", "llama2-70b-4096", "mixtral-8x7b-32768"))
    conversational_mem_len = st.sidebar.slider("Memory Length", 1, 10, 3)
    memory = ConversationBufferMemory(k=conversational_mem_len)

    user_input = st.text_area("Input: ")  

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    else:
        for message in st.session_state.chat_history:
            memory.save_context({"input": message['human']}, {"output": message['groq']})

    groq_chat = ChatGroq(api_key=groq_api, model=model)

    conversation = ConversationChain(llm=groq_chat, memory=memory)
    
    if user_input:
        response = conversation(user_input)
        message = {'human': user_input, 'groq': response['response']}
        st.session_state.chat_history.append(message)
        st.write(response['response'])


if __name__ == "__main__":
    main()
