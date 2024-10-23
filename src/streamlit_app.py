# Env import.
from os import getenv
from dotenv import load_dotenv

# Streamlit import.
import streamlit as st

# LangChain core imports.
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

# LangChain OpenAI and Pinecone imports.
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ------------------------------------------------------------------------------
# Loads the OPENAI_API_KEY from the .env file.
# Loads the PINECONE_API_KEY from the .env file.
load_dotenv()

# Retrieve the API keys from environment variables
openai_api_key = getenv('OPENAI_API_KEY')
openai_model_name = getenv('OPENAI_MODEL_NAME')
pinecone_api_key = getenv('PINECONE_API_KEY')
pintcone_index_name = getenv('PINECONE_INDEX_NAME')

# ------------------------------------------------------------------------------
# Initialize the OpenAI embeddings and the language model.
embeddings = OpenAIEmbeddings(api_key=openai_api_key)
llm = ChatOpenAI(model=openai_model_name, api_key=openai_api_key)

# ------------------------------------------------------------------------------
# Initialize the Pinecone client and the vector store.
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(name=pintcone_index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# ------------------------------------------------------------------------------
# Get the top relevant references from the Pinecone vector store.
def get_relevant_refs(query):
    refs = vector_store.similarity_search(
        query=query,
        k=5
    )
    return refs

# Get the response stream based on the user query.
def get_response_stream(query, chat_history):
    relevant_refs = get_relevant_refs(query)
    relevant_docs = [ref.page_content for ref in relevant_refs]
    template = '''
    You are an undergrad research advisor for the Computer Science and Engineering program at The Ohio State University. 
    You are helping students with their questions about general concepts, program adn career advise, and most importantly, research and faculty connections.
    You are given the chat history so far, the user question, and top relevant official info from the cse.osu.edu website.
    Be concise in your response, and prioritize the most up-to-date information.
    When appropriate, use bullet points to list multiple items.

    Chat history: {chat_history}

    User question: {user_question}

    Relevent info: {relevent_info}
    '''
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response_stream = chain.stream({
        'chat_history': chat_history,
        'user_question': query,
        'relevent_info': '\n'.join(relevant_docs)
    })
    ref_urls = [ref.metadata['url'] for ref in relevant_refs]
    return response_stream, ref_urls

# ------------------------------------------------------------------------------
# Init the chat history.
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------------------------------------------------
# Streamlit page config.
st.set_page_config(page_icon='🌰', page_title='Buck-AI-Guide')

# Streamlit page content.
st.title('🌰 💻 Buck-AI-Guide for CSE')
st.info('Hi! I am Buck-AI-Guide, your undergrad CSE research advisor. I know everything about the OSU CSE program. Ask me anything!')

# Print the converstation history.
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('Human'):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message('AI'):
             st.markdown(message.content)

# Get the next user query.
user_query = st.chat_input('your message here')
if user_query is not None and user_query != '':
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message('Human'):
        st.markdown(user_query)

    with st.chat_message('AI'):
        response_stream, ref_urls = get_response_stream(user_query, st.session_state.chat_history)
        ai_response = st.write_stream(response_stream)

        # Display the reference URLs.
        st.markdown('#### Official Reference')
        for url in ref_urls:
            st.markdown(f'* [{url}]({url})')

    st.session_state.chat_history.append(AIMessage(ai_response))
