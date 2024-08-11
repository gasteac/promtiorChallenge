import streamlit as st
# HumanMessage, AIMessage son clases que permiten definir mensajes de humanos y de IA
from langchain_core.messages import HumanMessage, AIMessage
# FAISS es una clase que permite indexar documentos en una tienda de vectores
from langchain_community.vectorstores import FAISS
## WebBaseLoader es una clase que permite cargar el texto de una página web (scrapping con beautifulSoup)
from langchain_community.document_loaders import WebBaseLoader
## RecursiveCharacterTextSplitter es una clase que permite dividir el texto en tokens (chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter
## OpenAIEmbeddings es una clase que permite cargar el modelo de lenguaje de OpenAI
from langchain_openai import OpenAIEmbeddings
## Para cargar el long language model de OpenAI
from langchain_openai import ChatOpenAI
## ChatPromptTemplate, MessagesPlaceholder son clases que permiten definir plantillas de prompts
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
## create_history_aware_retriever es una función que crea un recuperador de historial
from langchain.chains import create_history_aware_retriever
## create_stuff_documents_chain es una función que crea una cadena de documentos
from langchain.chains.combine_documents import create_stuff_documents_chain
## create_retrieval_chain es una función que crea una cadena de recuperación
from langchain.chains import create_retrieval_chain
## load_dotenv es una función que permite cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv
## Acá se cargan las variables de entorno
load_dotenv()

urls = [
    "https://www.promtior.ai",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/contacto",
    # "https://www.linkedin.com/company/promtior-ai/about/",
    # "https://www.instagram.com/promtior.ai/",
    "https://cie.ort.edu.uy/emprendimientos/promptior",
    "https://www.forbesuruguay.com/inteligencia-artificial/firma-uruguaya-asesora-empresas-ia-fue-elegida-100-startups-potenciarse-miami-n53538",
    "https://cuti.org.uy/destacados/la-startup-uruguaya-promtior-fue-seleccionada-ante-100-empresas-latinoamericanas-para-participar-del-programa-microsoft-ai-latino-en-miami/",
]

## Obtiene el texto de la página web
def get_vectorStore_from_urls(urls):
    loader = WebBaseLoader(urls)
    document = loader.load()
    ## Dividir el texto en tokens (chunks)
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    # Indexar los documentos en una tienda de vectores
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

## Este solo responde segun el contexto y la entrada del user
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    ## Las variables variable_name e input van a ser rellenadas con lo que sea que pase en la chain 
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ## Para que busque información en toda la conversación y no solo en el ultimo prompt
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation"),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

## Este esta mas mamado y responde segun el contexto, la entrada del user y el historial de chat :)
def get_conversational_rag_chain(retriever_chain):
    llm: ChatOpenAI = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based only on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_chain, stuff_document_chain)
    return retrieval_chain

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_query}
    )
    return response['answer']

## Ahora tenemos que unir nuestras 2 chains, la que tiene los documentos relevantes para nuestra conversación (retriever_chain)
## y la que usa ESOS documentos para entregar una respuesta dado el historial de chat también (retrieval_chain)

# App config
st.set_page_config(
    page_title="PromtChall",
    page_icon=":parrot:",
    initial_sidebar_state="expanded",
)
st.header("Ask anything about Promtior")
st.write("Made with ❤️ by Gasteac for [Promtior](https://www.promtior.ai/) 🤖")
st.divider()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]
    
# Display a spinner while loading documents
if "vector_store" not in st.session_state:
    with st.spinner("Please wait a moment, I'm loading a lot of documents 📚"):
        st.session_state.vector_store = get_vectorStore_from_urls(urls)
    
# User inputs
user_query = st.chat_input("Ask your question here")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    ##Guardamos en el historial la resp del usuario y de la ia
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

## Conversación, hago un loop en todos mis mensajes del historial del chat
for message in st.session_state.chat_history:
    ## Si es una instancia de AIMessage, muestro el mensaje con el nombre AI
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    ## Si es una instancia de HumanMessage, muestro el mensaje con el nombre You
    elif isinstance(message, HumanMessage):
        with st.chat_message("You"):
            st.write(message.content)
