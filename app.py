import streamlit as st

# HumanMessage, AIMessage son clases que permiten definir mensajes de humanos y de IA
from langchain_core.messages import HumanMessage, AIMessage

# FAISS es una clase que permite indexar documentos en una tienda de vectores
from langchain_community.vectorstores import FAISS

# WebBaseLoader es una clase que permite cargar el texto de una página web (scrapping con beautifulSoup)
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader

# RecursiveCharacterTextSplitter es una clase que permite dividir el texto en tokens (chunks)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAIEmbeddings es una clase que permite cargar el modelo de lenguaje de OpenAI
from langchain_openai import OpenAIEmbeddings

# Para cargar el long language model de OpenAI
from langchain_openai import ChatOpenAI

# Herramientas para el agente
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub

# Para cargar variables de entorno desde un archivo .env
from dotenv import load_dotenv

# Cargar las variables de entorno
load_dotenv()

# URLs web
urls = [
    "https://www.promtior.ai",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/contacto",
    "https://cie.ort.edu.uy/emprendimientos/promptior",
    "https://www.forbesuruguay.com/inteligencia-artificial/firma-uruguaya-asesora-empresas-ia-fue-elegida-100-startups-potenciarse-miami-n53538",
]


# Obtiene el texto de la página web o pdf, lo divide en tokens y lo indexa en una tienda de vectores
def get_vectorStore_from_sources(urls):
    documents = []
    # Cargar documentos desde URLs (HTML)
    web_loader = WebBaseLoader(urls)
    web_documents = web_loader.load()
    documents.extend(web_documents)

    # Cargar documento desde PDF
    pdf_loader = PyMuPDFLoader(
        "https://drive.google.com/uc?export=download&id=1-bHSg1UMDRqZTTGXEYpe2brPrLOeutqJ"
    )
    pdf_documents = pdf_loader.load()
    documents.extend(pdf_documents)

    # Dividir los documentos en tokens (chunks)
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    # Indexar los documentos en una tienda de vectores
    vector_store = FAISS.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


# Configuración de la app
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
        AIMessage(content="Hello, I'm here to solve your questions about Promtior :)"),
    ]

if "vector_store" not in st.session_state:
    with st.spinner("Please wait a moment, I'm loading a lot of documents 📚"):
        st.session_state.vector_store = get_vectorStore_from_sources(urls)

# Crear herramientas para el agente
retriever = st.session_state.vector_store.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "promtior_search",
    "Search for information about Promtior. For any questions about Promtior, you must use this tool!",
)

search = TavilySearchResults()
tools = [retriever_tool, search]

# Configurar el agente
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent_prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_openai_functions_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def get_response(user_input):
    response = agent_executor.invoke(
        {"input": user_input, "chat_history": st.session_state.chat_history}
    )
    return response["output"]


# Interfaz del chatbot
user_query = st.chat_input("Ask your question here")
if user_query:
    response = get_response(user_query)
    # Guardamos en el historial la respuesta del usuario y de la IA
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# Mostrar la conversación
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("You"):
            st.write(message.content)
