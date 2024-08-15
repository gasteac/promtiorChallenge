import streamlit as st

# HumanMessage, AIMessage son clases que permiten definir mensajes de humanos y de IA
from langchain_core.messages import HumanMessage, AIMessage

# FAISS es una clase que permite indexar documentos en una tienda de vectores
from langchain_community.vectorstores import FAISS

# WebBaseLoader es una clase que permite cargar el texto de una página web (scrapping con beautifulSoup)
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader

# RecursiveCharacterTextSplitter es una clase que permite dividir el texto (paginas grandes o pdfs o cosas muy grandes para el LLM) en chunks (cositas mas pequeñas)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# OpenAIEmbeddings es una clase que permite transformar a vectores los chunks (o mejor dicho su valor semántico)
from langchain_openai import OpenAIEmbeddings

# LLM de OpenAI
from langchain_openai import ChatOpenAI

# Herramientas para el agente

# Recuperador, busca vectores similares, tipo puntos en un gráfico 
from langchain.tools.retriever import create_retriever_tool
# Herramienta para el agente para que pueda acceder a internet (no lo necesito aca pero esta muy bueno)
from langchain_community.tools.tavily_search import TavilySearchResults
# El ejecutor del agente que va a utilizar al agente, y a las herramientas
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Para definir nuestro prompt para el agente, con mensajes de sistema, y placeholders (como el historial)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Traer variables de entorno del sistema, lo uso para las API keys en la instancia EC2 de AWS
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# URLs de Promtior o que contienen información relevante.
urls = [
    "https://www.promtior.ai",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
    "https://www.linkedin.com/company/promtior-ai/about/",
    # "https://infonegocios.biz/plus/con-tan-solo-un-ano-promtior-esta-bionizando-las-empresas-a-traves-de-la-adopcion-de-ia-generativa-y-se-expande-a-miami",
    # "https://cie.ort.edu.uy/emprendimientos/promptior",
    # "https://www.forbesuruguay.com/inteligencia-artificial/firma-uruguaya-asesora-empresas-ia-fue-elegida-100-startups-potenciarse-miami-n53538",
]

################## Esta función carga los documentos de las URLs y PDF's, los vuelve chunks, los transforma en vectores (valor semántico) y los indexa en una tienda de vectores (FAISS)
# Embedding es una representación de un texto en un espacio vectorial, donde cada palabra o frase es un vector....


def get_vectorStore_from_sources(urls):
    # Defino mi text splitter para transformar los documentos en chunks
    text_splitter = RecursiveCharacterTextSplitter()

    # Carga de paginas web
    web_loader = WebBaseLoader(urls)
    web_documents_chunks = web_loader.load_and_split(text_splitter)

    # Cargar de PDF's
    pdf_loader = PyMuPDFLoader(
        "https://drive.google.com/uc?export=download&id=1-bHSg1UMDRqZTTGXEYpe2brPrLOeutqJ"
    )
    pdf_documents_chunks = pdf_loader.load_and_split(text_splitter)

    # Las paginas web y los pdf's (ya tratados) se agregan a la lista de chunks
    chunks = pdf_documents_chunks + web_documents_chunks

    # Creamos la base de conocimiento con los vectores, OpenAIEmbeddings los transforma en vectores y luego se indexan
    vector_store = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return vector_store


################## Configuración de la app con Streamlit
st.set_page_config(
    page_title="PromtChall",
    page_icon=":parrot:",
    initial_sidebar_state="expanded",
)
st.header("Ask anything about Promtior")
# st.divider()

# Crear un historial de chat, y lo guardo en la sesión de Streamlit asi no se recarga cada vez que se actualiza la página (mando una pregunta)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm here to solve your questions about Promtior :)"),
    ]

# Lo mismo hago con la base de conocimiento, la guardo en la sesión de Streamlit para no tener que volver a cargarla cada vez que se actualiza la página
if "vector_store" not in st.session_state:
    # Mensajito alentador para que no se me vaya el usuario
    with st.spinner("Please wait a moment, I'm loading a lot of documents for you 📚"):
        st.session_state.vector_store = get_vectorStore_from_sources(urls)


# Retriever es una herramienta que busca documentos similares a un input en la bdc, ambos ya transformados a vectores.
retriever = st.session_state.vector_store.as_retriever()


################## Creamos el agente con las herramientas que le vamos a dar
def create_agent():
    # Le doy al agente una herramienta de búsqueda para encontrar información sobre Promtior en la bdc.
    retriever_tool = create_retriever_tool(
        retriever,
        "promtior_search",
        "Search for information about Promtior. For any questions about Promtior, you must use this tool!",
    )
    # Herramienta para que el agente pueda buscar en internet.
    search = TavilySearchResults()
    # Defino las herramientas que le voy a dar a mi agente.
    tools = [retriever_tool, search]

    ################## Configurar el agente, le damos un modelo y un prompt para responder preguntas.

    # Utilizamos el LLM de OpenAI
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # Prompt predefinido de OpenAI, sirve pero podemos hacer uno personalizado.
    # agent_prompt = hub.pull("hwchase17/openai-functions-agent")

    # Definimos como queremos que responda el agente, con mensajes de sistema, y placeholders para el historial y el scratchpad
    agent_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert representative of Promtior, a leading AI consulting firm.",
            ),
            (
                "system",
                "Your answers must be long and detailed.",
            ),
            # (
            #     "system",
            #     "You must only answer questions that are related to Promtior, they founders, personal, services, etc.",
            # ),
            (
                "system",
                "If somebody asks you about Promtior services, you should look in their website section 'services' and answer accordingly.",
            ),
            (
                "system",
                "If you feel you don't have enough information, use the talivy tool to search for information on the internet about everything related to Promtior.",
            ),
            (
                "system",
                "If a question is outside the scope of Promtior's domain, politely inform the user that you can only discuss topics related to Promtior.",
            ),
            # Queremos que el agente recuerde el historial del chat para poder responder mejor
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("user", "{input}"),
            (
                "system",
                "Use the below information to decide your next action, and then respond appropriately.",
            ),
            # TODO agent_scratchpad Creo que se utiliza para darle información sobre las herramientas que le proporcionamos y asi pueda utilizarlas.
            # Según la def original "Intermediate agent actions and tool output messages, will be passed in here."
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Crea un agente que utiliza el LLM de OpenAI para hacer llamadas a las funciones
    # Le pasamos un LLM que va a actuar como agente, las herramientas que le damos para que pueda buscar información y el prompt que le dice como responder
    agent = create_openai_functions_agent(llm, tools, agent_prompt)
    # TODO no entiendo quien es el agente al final el de arriba o el de abajo o los dos al mismo tiempo?
    # Agente que usa las herramientas, le pasamos el agente, las herramientas y verbose=False para que no muestre mensajes de debug
    return AgentExecutor(name="Raul", agent=agent, tools=tools, verbose=False)


# Si no existe un agente en la sesión de Streamlit, lo creamos
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = create_agent()


################## Función que le pasa la pregunta del usuario y el historial de chat al agente y retorna la respuesta del agente
def get_response(user_input):
    response = st.session_state.agent_executor.invoke(
        {"input": user_input, "chat_history": st.session_state.chat_history}
    )
    return response["output"]


################## Interfaz del chatbot con Streamlit

# Entrada de texto para que el usuario pueda hacer preguntas, la pregunta se guarda en user_query
user_query = st.chat_input("Ask your question here")
if user_query:
    # Obtenemos la respuesta del agente
    response = get_response(user_query)
    # Guardamos en el historial la respuesta del usuario y de la IA
    st.session_state.chat_history.extend(
        [HumanMessage(content=user_query), AIMessage(content=response)]
    )

################## Mostramos la conversación en pantalla

# AIMessage es una clase que representa los mensajes generados por la IA en la conversación
# HumanMessage es la clase que representa los mensajes enviados por el usuario
for message in st.session_state.chat_history:
    with st.chat_message("AI" if isinstance(message, AIMessage) else "You"):
        st.write(message.content)