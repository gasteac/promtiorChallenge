## Para ejecutar la aplicación correr: streamlit run app/app.py
# Clases para manejar diferentes tipos de mensajes en el chat: mensajes del sistema, del usuario y de la IA.
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# FAISS, una biblioteca para la búsqueda rápida de vectores, útil para manejar grandes cantidades de datos.
from langchain_community.vectorstores import FAISS
# Cargadores de documentos para cargar datos desde la web y archivos PDF.
from langchain_community.document_loaders import WebBaseLoader, PyMuPDFLoader
# Herramienta para dividir textos largos en fragmentos más pequeños.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Herramientas para trabajar con modelos de OpenAI, incluyendo embeddings y modelos de chat.
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Función para crear herramientas de recuperación de información (retrievers).
from langchain.tools.retriever import create_retriever_tool
# Función para crear un react agent preconstruido.
from langgraph.prebuilt import create_react_agent
# Clase para manejar el estado del agente de chat.
from langgraph.prebuilt.chat_agent_executor import AgentState
# Clase para guardar el estado del agente en memoria (historial de chat tmb).
from langgraph.checkpoint.memory import MemorySaver
# Streamlit es una biblioteca para crear aplicaciones web interactivas.
import streamlit as st
# Importa el módulo os para interactuar con las variables de entorno del archivo .env
import os
# Clases para manejar plantillas de prompts de chat.
from langchain_core.prompts import ChatPromptTemplate

# Variables de entorno para almacenar claves API y configuraciones
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Lista de URLs de Promtior desde donde se cargarán documentos para el vector store.
urls = [
    "https://www.promtior.ai",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
    "https://www.linkedin.com/company/promtior-ai/about/",
]

# Ruta del archivo PDF que se cargará.
path = "./docs/AIEngineer.pdf"

# Inicializa los embeddings de OpenAI, que son representaciones vectoriales de texto (de su semantica).
embedding = OpenAIEmbeddings()

# Función para obtener el vector store, donde se guardará la información vectorizada de los documentos.
def get_vectorStore_from_sources(urls):
    # Si el vector store ya existe, se carga desde el almacenamiento local.
    if os.path.exists("faiss_db"):
        vector_store = FAISS.load_local(
            "faiss_db", embedding, allow_dangerous_deserialization=True
        )
    # Si el vector store no existe, se crea a partir de los documentos.
    else:
        # Funcion que divide los textos en fragmentos más pequeños.
        text_splitter = RecursiveCharacterTextSplitter()
        # Carga documentos desde las URLs especificadas.
        web_loader = WebBaseLoader(urls)
        # Carga y divide los documentos web en fragmentos.
        web_documents_chunks = web_loader.load_and_split(text_splitter)
        # Carga el documento PDF.
        pdf_loader = PyMuPDFLoader(path)
        # Divide el documento PDF en fragmentos (chunks).
        pdf_documents_chunks = pdf_loader.load_and_split(text_splitter)
        # Combina los fragmentos (chunks) de documentos web y PDF.
        chunks = pdf_documents_chunks + web_documents_chunks
        # Crea el vector store a partir de los fragmentos de documentos.
        vector_store = FAISS.from_documents(chunks, embedding)
        # Guarda el vector store en el almacenamiento local.
        vector_store.save_local("faiss_db")
    return vector_store


# Configuración de la página de Streamlit
# Configura la página de Streamlit con un título, un ícono y un estado inicial de la barra lateral.
st.set_page_config(
    page_title="PromtChall",
    page_icon=":parrot:",
    initial_sidebar_state="expanded",
)
st.header("Ask anything about Promtior 🤖")
st.divider()
st.toast("Feel free to visit our website: https://www.promtior.ai", icon='🌐')
st.info("This is a RAG Chatbot created by Gaston Acosta, AI Engineer at Promtior.")

# Inicializar el historial de chat
# Si el historial de chat no existe en el estado de la sesión, se inicializa con un mensaje de bienvenida de la IA.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm here to solve your questions about Promtior :)"),
    ]

# Cargar el vector store
# Si el vector store no existe en el estado de la sesión, se carga y se muestra un spinner mientras se realiza la carga.
if "vector_store" not in st.session_state:
    with st.spinner("Please wait a moment, I'm loading a lot of documents for you 📚"):
        st.session_state.vector_store = get_vectorStore_from_sources(urls)

# Crea una herramienta de recuperación de información a partir del vector store.
retriever = st.session_state.vector_store.as_retriever()

# Crear y configurar el agente
# Función para modificar el estado del agente, invocando la plantilla de prompt con los mensajes actuales.
def _modify_state_messages(state: AgentState):
    return prompt.invoke({"messages": state["messages"]}).to_messages()


# Crea una herramienta de recuperación de información específica para buscar información sobre Promtior.
retriever_tool = create_retriever_tool(
    retriever, "promtior_search", "Search for information about Promtior."
)

# Lista de herramientas que el agente puede usar, podria agregar Tavily pero no quiero.
tools = [retriever_tool]

system_message = """
You are an expert assistant of Promtior, a leading AI consulting firm.
Answer ONLY based on the documents you have.
Your answers must be long and detailed.
Gaston Acosta is your creator (always mention it if somebody asks about him), and he works at Promtior as an AI Engineer.
If somebody asks you about Promtior services, you should look in their website section 'services' and answer accordingly.
If a question is outside the scope of Promtior's domain, politely inform the user that you can only discuss topics related to Promtior.
Do not answer any questions that are not directly related to Promtior. If a question is not related to Promtior, respond with: "I can only answer questions related to Promtior."
"""

# Define una plantilla de prompt para el chat, especificando cómo debe comportarse el agente y qué información debe proporcionar.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        ("placeholder", "{messages}"),
    ]
)

# Inicializa un objeto para guardar el estado del agente en memoria.
memory = MemorySaver()

# Inicializa el modelo de chat de OpenAI con el modelo "gpt-4o" y una temperatura de 0 (respuestas determinísticas).
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Crea el react agent utilizando el modelo, las herramientas, el modificador de estado y el guardado en memoria.
app = create_react_agent(
    model, tools, state_modifier=_modify_state_messages, checkpointer=memory
)

# Configuración adicional para el agente, para que recuerde el hilo de chat.
# La clave "thread_id" se utiliza para identificar el hilo de chat,
# lo que permite al agente recordar y mantener el contexto de la conversación.
config = {"configurable": {"thread_id": "test-thread"}}


# Obtener la respuesta del agente
def get_response(user_input):
    # Preparar los mensajes para el agente, incluyendo el historial de chat y el mensaje del usuario actual.
    # Lo obtenemos del estado de la sesión de Streamlit.
    # Sino el agente no recuerda lo que hablamos.
    messages = [
        ("human", msg.content) if isinstance(msg, HumanMessage) else ("ai", msg.content)
        for msg in st.session_state.chat_history
    ]
    
    # Convierte el historial de chat en una lista de tuplas y añade el mensaje del usuario actual.
    messages.append(("human", user_input))

    # Obtener la respuesta del agente
    # Invoca el agente con los mensajes preparados y la configuración para recordar el hilo del chat.
    response = app.invoke({"messages": messages}, config)

    # Devuelve el contenido del último mensaje de la respuesta del agente.
    return response["messages"][-1].content


# Interfaz de usuario de Streamlit

# Muestra un campo de entrada de chat para que el usuario ingrese su pregunta.
user_query = st.chat_input("Write your questions here :)")

# Si el usuario ingresa una pregunta:
if user_query:
    response = get_response(user_query)
    # Añade el mensaje del usuario y la respuesta de la IA al historial de chat.
    st.session_state.chat_history.extend(
        [HumanMessage(content=user_query), AIMessage(content=response)]
    )

# Itera sobre el historial de chat y muestra cada mensaje en la interfaz de usuario, diferenciando entre mensajes de la IA y del usuario.
for message in st.session_state.chat_history:
    # Muestra el mensaje de la IA con un indicador "AI" y el mensaje del usuario con un indicador "You".
    # Pregunto si el mensaje devuelto por AI o el escrito por You son instancias de la clase AIMessage o HumanMessage.
    # Si el mensaje es de la clase AI, muestra en forma de "AI" y si es de la clase Human, muestra en forma de "You".
    with st.chat_message("AI" if isinstance(message, AIMessage) else "You"):
        # Muestra el contenido del mensaje en el chat.
        # El simbolo de AI o de You se muestra automaticamente dependiendo de la clase del mensaje.
        st.write(message.content)
