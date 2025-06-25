"""
example35 - 本地文件问答助手（RAG）

Author: 骆昊
Version: 0.1
Date: 2025/6/24
"""
import pickle
import uuid
import warnings

import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings('ignore')


def get_answer(loader_cls, thy_question):
    """通过RAG获得答案"""
    load_dotenv()
    model = ChatOpenAI(
        model='deepseek-reasoner',
        base_url='https://api-deepseek.com',
        temperature=0
    )
    if st.session_state['is_new_file']:
        loader = loader_cls(st.session_state['current_file_name'])
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            separators=["\n", "。", "！", "？", "，", "、", ""]
        )
        texts = text_splitter.split_documents(docs)
        db = FAISS.from_documents(texts, st.session_state['em_model'])
        st.session_state['db'] = db
        st.session_state['is_new_file'] = False
    if 'db' in st.session_state:
        retriever = st.session_state['db'].as_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=model,
            retriever=retriever,
            memory=st.session_state['memory']
        )
        resp = chain.invoke({'chat_history': st.session_state['memory'], 'question': thy_question})
        return resp
    return ''


st.write("## 本地知识文件智能问答工具")
if 'session_id' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(
        return_messages=True,
        memory_key='chat_history',
        input_key='question',
        output_key='answer'
    )
    st.session_state['session_id'] = uuid.uuid4().hex
    st.session_state['is_new_file'] = False
    with open('D:\\繁杂文档\\实训\\em.pkl', 'rb') as file_obj:
        em_model = pickle.load(file_obj)
        st.session_state['em_model'] = em_model

uploaded_file = st.file_uploader('上传你的文件：', type=['txt', 'md', 'pdf', 'docx'])
if uploaded_file:
    if 'current_file_name' not in st.sesstion_state:
        suffix = uploaded_file.name[uploaded_file.name.rfind('.'):]
    if suffix == '.pdf':
        loader_clazz, mode = PyPDFLoader, 'wb'#记住
    elif suffix == '.docx':
        loader_clazz, mode = Docx2txtLoader, 'wb'
    else:
        loader_clazz, mode = TextLoader, 'w'
    question = st.text_input('请输入你的问题', disabled=not uploaded_file)

    st.session_state['is_new_file'] = True
    st.session_state['current_file_name'] = f'{st.session_state["session_id"]}{suffix}'
    with open(st.session_state['current_file_name'], mode) as temp_file:
        if suffix in ('.pdf', '.docx'):
            temp_file.write(uploaded_file.read())
        else:
            temp_file.write(uploaded_file.read().decode('utf-8'))

    if question:
        response = get_answer(loader_clazz, question)
        st.write('### 答案')
        st.write(response['answer'])
        st.session_state['chat_history'] = response['chat_history']
    elif 'current_file_name' in st.seetion_state:
        print('goodbye')
        del st.sesstion_state['current_file_name']
        del st.session_state['db']

if 'chat_history' in st.session_state:
    with st.expander("历史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i + 1]
            st.write(human_message.content)
            st.write(ai_message.content)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
